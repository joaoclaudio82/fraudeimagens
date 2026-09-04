"""Utilidades compartilhadas pelos testes: Tesseract falso e palavras de um comprovante."""

import sys
import types


def fake_tesseract(monkeypatch, words, fail_data=False, fail_all=False):
    """Instala um pytesseract falso em sys.modules: devolve as palavras informadas sem precisar do binário.

    ``words`` é uma lista de linhas; cada linha é uma lista de dicts com text, left, top, width, height
    e, opcionalmente, conf. Devolve um contador de chamadas para verificar fallbacks.
    """
    module = types.ModuleType("pytesseract")
    module.Output = types.SimpleNamespace(DICT="dict")
    calls = {"data": 0, "string": 0}

    def image_to_data(image, lang, config, output_type):
        calls["data"] += 1
        if fail_data or fail_all:
            raise RuntimeError("tesseract not found")
        keys = ["text", "conf", "left", "top", "width", "height", "block_num", "par_num", "line_num"]
        data = {key: [] for key in keys}
        for line_no, line in enumerate(words):
            for word in line:
                data["text"].append(word["text"])
                data["conf"].append(word.get("conf", 90))
                data["left"].append(word["left"])
                data["top"].append(word["top"])
                data["width"].append(word["width"])
                data["height"].append(word["height"])
                data["block_num"].append(1)
                data["par_num"].append(1)
                data["line_num"].append(line_no + 1)
        return data

    def image_to_string(image, lang, config):
        calls["string"] += 1
        if fail_all:
            raise RuntimeError("tesseract not found")
        return "\n".join(" ".join(w["text"] for w in line) for line in words)

    module.image_to_data = image_to_data
    module.image_to_string = image_to_string
    monkeypatch.setitem(sys.modules, "pytesseract", module)
    return calls


def text_line(y, *texts, height=40, char_width=30, start=100):
    """Linha de palavras com caixas consecutivas (coordenadas na escala 2x usada pelo OCR)."""
    x, out = start, []
    for text in texts:
        out.append({"text": text, "left": x, "top": y, "width": char_width * len(text), "height": height})
        x += char_width * len(text) + 20
    return out


def receipt_words():
    return [
        text_line(100, "COMPROVANTE", "DE", "ENTREGA"),
        text_line(160, "PEDIDO", "12345"),
        text_line(220, "DATA", "04/09/2026"),
        text_line(280, "DESTINATARIO", "JOAO", "CLAUDIO"),
        text_line(340, "VALOR", "R$", "1.250,00"),
        text_line(400, "CODIGO", "ABC-7781"),
    ]
