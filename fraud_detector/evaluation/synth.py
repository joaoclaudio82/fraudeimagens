"""Comprovantes sintéticos e forjarias controladas.

Serve aos testes automatizados e ao harness de avaliação: como sabemos exatamente
onde cada edição foi feita, dá para medir se um sinal aponta a região certa, e não
apenas se o score subiu.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_SIZE = 28
LINE_HEIGHT = 60
TEXT_X = 140
TEXT_Y0 = 160
PAPER_BOX = (100, 100, 1100, 800)
LINE_INDEX = {"titulo": 0, "pedido": 1, "data": 2, "destinatario": 3, "valor": 4, "codigo": 5}
SOFTWARE_TAG = 0x0131
MAKE_TAG = 0x010F
DATETIME_TAG = 0x0132
DATETIME_ORIGINAL_TAG = 0x9003
EXIF_IFD_TAG = 0x8769


@dataclass(frozen=True)
class ReceiptSpec:
    order_id: str = "12345"
    date: str = "04/09/2026"
    recipient: str = "JOAO CLAUDIO"
    value: str = "R$ 1.250,00"
    code: str = "ABC-7781"
    seed: int = 0

    def lines(self) -> list[tuple[str, str]]:
        return [
            ("titulo", "COMPROVANTE DE ENTREGA"),
            ("pedido", f"PEDIDO {self.order_id}"),
            ("data", f"DATA {self.date}"),
            ("destinatario", f"DESTINATARIO {self.recipient}"),
            ("valor", f"VALOR {self.value}"),
            ("codigo", f"CODIGO {self.code}"),
        ]


def line_region(line: str, width: int = 700) -> dict[str, int]:
    index = LINE_INDEX[line]
    return {"x": TEXT_X - 10, "y": TEXT_Y0 + index * LINE_HEIGHT - 8, "w": width, "h": 48}


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=FONT_SIZE)
    except TypeError:  # Pillow antigo sem fonte vetorial embutida
        return ImageFont.load_default()


def _paper(rng: np.random.Generator, width: int, height: int, base: float = 235.0, noise: float = 4.0) -> Image.Image:
    sheet = np.clip(base + rng.normal(0, noise, (height, width)), 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([sheet] * 3, axis=-1))


def render_receipt(spec: ReceiptSpec = ReceiptSpec(), size: tuple[int, int] = (1200, 900)) -> Image.Image:
    """Foto simulada: fundo com gradiente e ruído de sensor, folha texturizada e texto impresso."""
    rng = np.random.default_rng(spec.seed)
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    phase = rng.uniform(0, 6.28, 2)
    base = 120 + 60 * np.sin(xx / 90 + phase[0]) + 40 * np.cos(yy / 70 + phase[1])
    background = np.clip(base + rng.normal(0, 6, (height, width)), 0, 255)
    tint = rng.uniform(0.85, 1.0, 3)
    image = Image.fromarray(np.clip(np.stack([background * tint[i] for i in range(3)], axis=-1), 0, 255).astype(np.uint8))

    x0, y0, x1, y1 = PAPER_BOX
    image.paste(_paper(rng, x1 - x0, y1 - y0), (x0, y0))
    draw = ImageDraw.Draw(image)
    font = _font()
    for index, (_, text) in enumerate(spec.lines()):
        draw.text((TEXT_X, TEXT_Y0 + index * LINE_HEIGHT), text, fill=(20, 20, 20), font=font)
    points = [(TEXT_X + 40 * i + int(rng.integers(-10, 10)), 560 + int(rng.integers(-25, 25))) for i in range(12)]
    draw.line(points, fill=(30, 30, 90), width=3)
    draw.text((TEXT_X, 610), "ASSINATURA", fill=(20, 20, 20), font=font)
    return image


def open_rgb(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def jpeg_bytes(image: Image.Image, quality: int = 90, exif: bytes | None = None) -> bytes:
    buffer = io.BytesIO()
    kwargs = {"quality": int(quality)}
    if exif:
        kwargs["exif"] = exif
    image.convert("RGB").save(buffer, "JPEG", **kwargs)
    return buffer.getvalue()


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def double_compressed(image: Image.Image, first_quality: int = 75, second_quality: int = 90) -> bytes:
    """Cadeia típica: câmera/WhatsApp salva em q≈75 e um novo salvamento em q≈90, sem edição."""
    return jpeg_bytes(open_rgb(jpeg_bytes(image, first_quality)), second_quality)


def forge_value_edit(data: bytes, new_text: str = "VALOR R$ 9.250,00", line: str = "valor",
                     quality: int = 90, seed: int = 1) -> tuple[bytes, dict[str, int]]:
    """Reescreve uma linha sobre papel novo, como faria alguém editando o valor em um editor."""
    rng = np.random.default_rng(seed)
    image = open_rgb(data)
    region = line_region(line)
    image.paste(_paper(rng, region["w"], region["h"]), (region["x"], region["y"]))
    ImageDraw.Draw(image).text((TEXT_X, TEXT_Y0 + LINE_INDEX[line] * LINE_HEIGHT), new_text,
                               fill=(20, 20, 20), font=_font())
    return jpeg_bytes(image, quality), region


def forge_copy_move(data: bytes, quality: int = 90) -> tuple[bytes, dict[str, int], dict[str, int]]:
    """Copia o bloco do número do pedido e cola no fim da linha do código (mesma imagem)."""
    image = open_rgb(data)
    source = {"x": TEXT_X, "y": TEXT_Y0 + LINE_INDEX["pedido"] * LINE_HEIGHT - 6, "w": 300, "h": 44}
    target = {"x": TEXT_X + 420, "y": TEXT_Y0 + LINE_INDEX["codigo"] * LINE_HEIGHT - 6, "w": 300, "h": 44}
    patch = image.crop((source["x"], source["y"], source["x"] + source["w"], source["y"] + source["h"]))
    image.paste(patch, (target["x"], target["y"]))
    return jpeg_bytes(image, quality), source, target


def forge_splice(data_a: bytes, data_b: bytes, line: str = "valor", quality: int = 90) -> tuple[bytes, dict[str, int]]:
    """Cola a linha de outro comprovante (outra foto, outra compressão) sobre este."""
    target, donor = open_rgb(data_a), open_rgb(data_b)
    region = line_region(line)
    box = (region["x"], region["y"], region["x"] + region["w"], region["y"] + region["h"])
    target.paste(donor.crop(box), (region["x"], region["y"]))
    return jpeg_bytes(target, quality), region


def with_exif(data: bytes, software: str | None = None, datetime_original: str | None = None,
              datetime_modified: str | None = None, make: str | None = None, quality: int = 92,
              qtables: list[list[int]] | None = None) -> bytes:
    """Regrava o JPEG com tags EXIF (datas no formato 'AAAA:MM:DD HH:MM:SS') e, opcionalmente, tabelas próprias."""
    image = Image.open(io.BytesIO(data))
    exif = image.getexif()
    if software:
        exif[SOFTWARE_TAG] = software
    if make:
        exif[MAKE_TAG] = make
    if datetime_modified:
        exif[DATETIME_TAG] = datetime_modified
    if datetime_original:
        exif.get_ifd(EXIF_IFD_TAG)[DATETIME_ORIGINAL_TAG] = datetime_original
    buffer = io.BytesIO()
    kwargs: dict = {"quality": int(quality), "exif": exif.tobytes()}
    if qtables:
        kwargs["qtables"] = qtables
    image.convert("RGB").save(buffer, "JPEG", **kwargs)
    return buffer.getvalue()


def whatsapp_like(data: bytes, quality: int = 70, max_side: int = 1280) -> bytes:
    """Redimensiona, recomprime e remove metadados, como apps de mensagem fazem."""
    image = open_rgb(data)
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return jpeg_bytes(image, quality)


def screenshot_like(data: bytes, screen: tuple[int, int] = (1080, 1920)) -> bytes:
    """Captura de tela: imagem centralizada em um 'celular' com barras escuras, salva em PNG."""
    image = open_rgb(data)
    canvas = Image.new("RGB", screen, (18, 18, 18))
    scale = screen[0] / image.width
    fitted = image.resize((screen[0], round(image.height * scale)), Image.Resampling.LANCZOS)
    canvas.paste(fitted, (0, (screen[1] - fitted.height) // 2))
    return png_bytes(canvas)


def blurred(data: bytes, radius: float = 4.0, quality: int = 85) -> bytes:
    return jpeg_bytes(open_rgb(data).filter(ImageFilter.GaussianBlur(radius)), quality)


# ---------------------------------------------------------------------------
# Base rotulada
# ---------------------------------------------------------------------------

FIRST_NAMES = ["ANA", "BRUNO", "CARLA", "DIEGO", "ELISA", "FABIO", "GISELE", "HUGO", "IARA", "JOAO", "KARLA", "LUCAS"]
LAST_NAMES = ["SILVA", "SOUZA", "OLIVEIRA", "LIMA", "PEREIRA", "COSTA", "ROCHA", "ALMEIDA", "NUNES", "CARVALHO"]
INTACT_ORIGINS = ["camera", "camera", "double", "whatsapp", "screenshot", "blur"]
FORGED_KINDS = ["value_edit", "value_edit", "copy_move", "splice", "editor_exif", "reuse", "date_mismatch"]


def random_spec(rng: np.random.Generator, seed: int) -> ReceiptSpec:
    day = int(rng.integers(1, 28))
    month = int(rng.integers(1, 12))
    value = f"R$ {int(rng.integers(1, 9999)):,}".replace(",", ".") + f",{int(rng.integers(0, 99)):02d}"
    return ReceiptSpec(
        order_id=str(int(rng.integers(10000, 99999))),
        date=f"{day:02d}/{month:02d}/2026",
        recipient=f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
        value=value,
        code=f"{rng.choice(list('ABCDEFGH'))}{rng.choice(list('JKLMNP'))}{rng.choice(list('QRSTUV'))}-{int(rng.integers(1000, 9999))}",
        seed=seed,
    )


def expected_fields(spec: ReceiptSpec) -> dict[str, str]:
    return {"pedido": spec.order_id, "data": spec.date, "destinatario": spec.recipient,
            "valor": spec.value, "codigo": spec.code}


def _exif_date(spec: ReceiptSpec, hour: int = 10, shift_days: int = 0) -> str:
    day, month, year = (int(part) for part in spec.date.split("/"))
    day = max(1, min(28, day + shift_days))
    return f"{year:04d}:{month:02d}:{day:02d} {hour:02d}:00:00"


def camera_like(image: Image.Image, spec: ReceiptSpec, rng: np.random.Generator) -> bytes:
    """Foto original: EXIF de câmera com firmware em Software e data de captura no dia da entrega."""
    quality = int(rng.integers(85, 96))
    firmware = str(rng.choice(["17.5.1", "S918BXXU3CWL1", "HDR+ 1.0.345", "Google"]))
    make = str(rng.choice(["Apple", "samsung", "Xiaomi", "motorola"]))
    return with_exif(jpeg_bytes(image, quality), software=firmware, make=make,
                     datetime_original=_exif_date(spec, int(rng.integers(8, 19))), quality=quality)


def make_intact(spec: ReceiptSpec, origin: str, rng: np.random.Generator) -> bytes:
    image = render_receipt(spec)
    photo = camera_like(image, spec, rng)
    if origin == "camera":
        return photo
    if origin == "double":
        return double_compressed(image, int(rng.integers(70, 86)), int(rng.integers(88, 96)))
    if origin == "whatsapp":
        return whatsapp_like(photo, quality=int(rng.integers(65, 80)))
    if origin == "screenshot":
        return screenshot_like(photo)
    if origin == "blur":
        return blurred(photo, radius=float(rng.uniform(2.5, 5.0)))
    raise ValueError(origin)


def make_forged(spec: ReceiptSpec, kind: str, rng: np.random.Generator, donor: ReceiptSpec,
                previous: bytes | None) -> tuple[bytes, dict[str, Any] | None, dict[str, str]]:
    """Devolve (bytes, região editada ou None, campos esperados pelo sistema)."""
    image = render_receipt(spec)
    base = jpeg_bytes(image, int(rng.integers(72, 90)))
    expected = expected_fields(spec)
    out_quality = int(rng.integers(86, 96))
    if kind == "value_edit":
        new_value = f"R$ {int(rng.integers(1000, 9999))},{int(rng.integers(0, 99)):02d}"
        data, region = forge_value_edit(base, new_text=f"VALOR {new_value}", quality=out_quality, seed=spec.seed + 1)
        return data, region, expected
    if kind == "copy_move":
        data, _, target = forge_copy_move(base, quality=out_quality)
        return data, target, expected
    if kind == "splice":
        donor_data = jpeg_bytes(render_receipt(donor), int(rng.integers(60, 80)))
        data, region = forge_splice(base, donor_data, quality=out_quality)
        return data, region, expected
    if kind == "editor_exif":
        data = with_exif(base, software=str(rng.choice(["Adobe Photoshop 25.0 (Windows)", "GIMP 2.10.34", "Snapseed 2.0"])),
                         datetime_original=_exif_date(spec, 9), datetime_modified=_exif_date(spec, 14), quality=out_quality)
        return data, None, expected
    if kind == "reuse":
        # Mesma foto de uma entrega anterior, reenviada por WhatsApp para outro pedido.
        source = previous or camera_like(image, spec, rng)
        other = random_spec(rng, spec.seed + 7)
        return whatsapp_like(source, quality=int(rng.integers(65, 80))), None, expected_fields(other)
    if kind == "date_mismatch":
        shift = int(rng.choice([-5, -3, -2, 2, 4]))
        data = with_exif(base, software="17.5.1", make="Apple", datetime_original=_exif_date(spec, 11, shift), quality=out_quality)
        return data, None, expected
    raise ValueError(kind)


def generate_dataset(out_dir: str | Path, n_intact: int = 30, n_forged: int = 30, seed: int = 0,
                     whatsapp_fraction: float = 0.3,
                     progress: Callable[[int, int], None] | None = None) -> list[dict[str, Any]]:
    """Gera imagens + labels.jsonl. Cada item registra rótulo, tipo, origem, região editada e campos esperados."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    items: list[dict[str, Any]] = []
    total = n_intact + n_forged
    previous_photo: bytes | None = None

    for index in range(n_intact):
        spec = random_spec(rng, seed * 1000 + index)
        origin = INTACT_ORIGINS[index % len(INTACT_ORIGINS)]
        data = make_intact(spec, origin, rng)
        if origin == "camera":
            previous_photo = data
        name = f"{index:04d}_intact_{origin}.{'png' if origin == 'screenshot' else 'jpg'}"
        (out / name).write_bytes(data)
        items.append({"file": name, "label": 0, "kind": "intact", "origin": origin, "region": None,
                      "expected": expected_fields(spec), "spec": asdict(spec)})
        if progress:
            progress(len(items), total)

    for index in range(n_forged):
        spec = random_spec(rng, seed * 1000 + 500 + index)
        donor = random_spec(rng, seed * 1000 + 700 + index)
        kind = FORGED_KINDS[index % len(FORGED_KINDS)]
        data, region, expected = make_forged(spec, kind, rng, donor, previous_photo)
        origin = "direct"
        if kind not in ("reuse",) and rng.random() < whatsapp_fraction:
            data = whatsapp_like(data, quality=int(rng.integers(65, 80)))
            origin = "whatsapp"
            if region:
                scale = 1280 / 1200
                region = {k: int(round(v * scale)) for k, v in region.items()}
        name = f"{n_intact + index:04d}_forged_{kind}_{origin}.jpg"
        (out / name).write_bytes(data)
        items.append({"file": name, "label": 1, "kind": kind, "origin": origin, "region": region,
                      "expected": expected, "spec": asdict(spec)})
        if progress:
            progress(len(items), total)

    with (out / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return items
