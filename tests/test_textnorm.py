from datetime import date
from decimal import Decimal

import pytest

from fraud_detector.textnorm import (
    find_dates, find_money, fix_digit_confusions, infer_kind, match_field, normalize_code, parse_date, parse_money,
)

OCR_TEXT = (
    "COMPROVANTE DE ENTREGA\nPedido: 12345\nData: 04/09/2026\nDestinatario: JOAO CLAUDIO\n"
    "VALOR R$1.250,00\nCod. 0O1-A8Z"
)


@pytest.mark.parametrize("field, value", [
    ("pedido", "12345"),
    ("data", "04-09-2026"),
    ("data", "2026-09-04"),
    ("destinatário", "João Cláudio"),
    ("valor", "R$ 1.250,00"),
    ("código", "001-A8Z"),
])
def test_realistic_ocr_variations_match(field, value):
    assert match_field(field, value, OCR_TEXT).matched, (field, value)


def test_real_divergences_are_not_forgiven():
    other_date = match_field("data", "05/09/2026", OCR_TEXT)
    assert not other_date.matched and other_date.conflict and other_date.found == "2026-09-04"

    other_value = match_field("valor", "R$ 9.250,00", OCR_TEXT)
    assert not other_value.matched and other_value.conflict and other_value.found == "1250.00"

    other_order = match_field("pedido", "12399", OCR_TEXT)
    assert not other_order.matched and other_order.conflict


def test_missing_versus_conflict():
    missing = match_field("valor", "R$ 10,00", "texto sem numeros")
    assert not missing.matched and not missing.conflict and missing.found is None


@pytest.mark.parametrize("text, expected", [
    ("04/09/2026", date(2026, 9, 4)),
    ("4/9/26", date(2026, 9, 4)),
    ("2026-09-04", date(2026, 9, 4)),
    ("04.09.2026", date(2026, 9, 4)),
    ("4 de setembro de 2026", date(2026, 9, 4)),
    ("04 set 2026", date(2026, 9, 4)),
    ("2026-09-04T10:15:00", date(2026, 9, 4)),
    ("O4/O9/2026", date(2026, 9, 4)),
    ("31/02/2026", None),
    ("sem data", None),
])
def test_parse_date(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("R$ 1.250,00", Decimal("1250.00")),
    ("R$1250,00", Decimal("1250.00")),
    ("1250.00", Decimal("1250.00")),
    ("1.250", Decimal("1250.00")),
    ("12,5", Decimal("12.50")),
    ("1,250.00", Decimal("1250.00")),
    ("abc", None),
])
def test_parse_money(text, expected):
    assert parse_money(text) == expected


def test_find_dates_and_money_in_noisy_text():
    text = "Entregue em 04/09/2026 as 10:15. Total R$ 1.250,00. Frete 12,50. Tel 85999990000"
    assert date(2026, 9, 4) in find_dates(text)
    money = find_money(text)
    assert Decimal("1250.00") in money and Decimal("12.50") in money
    assert Decimal("85999990000.00") not in money


def test_digit_confusions_only_in_numeric_tokens():
    assert fix_digit_confusions("Pedido 1O0O5 OLIVEIRA") == "Pedido 10005 OLIVEIRA"
    assert normalize_code("0O1-A8Z") == normalize_code("001-A8Z")


def test_kind_inference():
    assert infer_kind("data", "04/09/2026") == "date"
    assert infer_kind("qualquer", "04/09/2026") == "date"
    assert infer_kind("valor total", "10,00") == "money"
    assert infer_kind("pedido", "12345") == "number"
    assert infer_kind("rastreio", "BR123456789") == "code"
    assert infer_kind("destinatario", "Maria Silva") == "text"
