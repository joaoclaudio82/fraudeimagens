"""Comprovantes sintéticos e forjarias controladas.

Serve aos testes automatizados e ao harness de avaliação: como sabemos exatamente
onde cada edição foi feita, dá para medir se um sinal aponta a região certa, e não
apenas se o score subiu.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

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
