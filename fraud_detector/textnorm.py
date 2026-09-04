"""Normalização e casamento tolerante entre o texto do OCR e os campos esperados.

O OCR erra de formas previsíveis (0/O, 1/l, acentos perdidos, espaços a mais) e
os sistemas escrevem o mesmo dado de formas diferentes (04/09/2026 vs 2026-09-04,
"R$ 1.250,00" vs "1250.00"). Comparar substring exata gera falso positivo em
comprovante legítimo; comparar valor semântico (data, dinheiro, número) evita
isso sem deixar passar uma divergência real.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from rapidfuzz import fuzz

MONTHS = {
    "jan": 1, "janeiro": 1, "fev": 2, "fevereiro": 2, "mar": 3, "marco": 3, "abr": 4, "abril": 4,
    "mai": 5, "maio": 5, "jun": 6, "junho": 6, "jul": 7, "julho": 7, "ago": 8, "agosto": 8,
    "set": 9, "setembro": 9, "out": 10, "outubro": 10, "nov": 11, "novembro": 11, "dez": 12, "dezembro": 12,
}
CODE_CONFUSIONS = str.maketrans({"O": "0", "Q": "0", "I": "1", "L": "1", "|": "1", "Z": "2", "S": "5", "B": "8"})
DIGIT_CONFUSIONS = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"})

DATE_NUMERIC = re.compile(r"(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})(?!\d)")
DATE_ISO = re.compile(r"(?<!\d)(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})(?!\d)")
DATE_WRITTEN = re.compile(r"(?<!\d)(\d{1,2})\s*(?:de\s+)?([a-z]{3,9})\.?\s*(?:de\s+)?(\d{2,4})(?!\d)")
MONEY = re.compile(r"(?:r\$\s*)?(?<![\d,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{2}))?(?!\d|\.\d)")
DIGIT_RUN = re.compile(r"\d{3,}")
NUMERIC_TOKEN = re.compile(r"[0-9OoIl|][0-9OoIl|/.,:\-]*[0-9OoIl|]")


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    text = strip_accents(text).casefold()
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_code(text: str) -> str:
    text = strip_accents(text).upper()
    text = re.sub(r"[^0-9A-Z|]", "", text)
    return text.translate(CODE_CONFUSIONS)


def fix_digit_confusions(text: str) -> str:
    """Troca O/I/l por 0/1 apenas dentro de tokens que são majoritariamente numéricos."""

    def repair(match: re.Match[str]) -> str:
        token = match.group(0)
        digits = sum(ch.isdigit() for ch in token)
        letters = sum(ch in "OoIl|" for ch in token)
        if digits >= 2 and digits >= letters:
            return token.translate(DIGIT_CONFUSIONS)
        return token

    return NUMERIC_TOKEN.sub(repair, text)


def _year(value: str) -> int:
    year = int(value)
    return year + 2000 if year < 100 else year


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date(text: str) -> date | None:
    cleaned = fix_digit_confusions(strip_accents(text).strip().casefold())
    if match := DATE_ISO.fullmatch(cleaned):
        return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if match := DATE_NUMERIC.fullmatch(cleaned):
        return _safe_date(_year(match.group(3)), int(match.group(2)), int(match.group(1)))
    if match := DATE_WRITTEN.fullmatch(cleaned):
        month = MONTHS.get(match.group(2))
        return _safe_date(_year(match.group(3)), month, int(match.group(1))) if month else None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def find_dates(text: str) -> list[date]:
    cleaned = fix_digit_confusions(strip_accents(text).casefold())
    found: list[date] = []
    for match in DATE_ISO.finditer(cleaned):
        if parsed := _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3))):
            found.append(parsed)
    for match in DATE_NUMERIC.finditer(cleaned):
        if parsed := _safe_date(_year(match.group(3)), int(match.group(2)), int(match.group(1))):
            found.append(parsed)
    for match in DATE_WRITTEN.finditer(cleaned):
        month = MONTHS.get(match.group(2))
        if month and (parsed := _safe_date(_year(match.group(3)), month, int(match.group(1)))):
            found.append(parsed)
    return found


def parse_money(text: str) -> Decimal | None:
    cleaned = fix_digit_confusions(strip_accents(text)).casefold().replace("r$", "").replace(" ", "")
    if not re.fullmatch(r"[\d.,]+", cleaned) or not any(ch.isdigit() for ch in cleaned):
        return None
    if "," in cleaned and "." in cleaned:
        decimal_sep = "," if cleaned.rfind(",") > cleaned.rfind(".") else "."
    elif "," in cleaned:
        decimal_sep = ","
    elif "." in cleaned:
        tail = cleaned.rsplit(".", 1)[1]
        decimal_sep = "." if len(tail) == 2 and cleaned.count(".") == 1 else None
    else:
        decimal_sep = None
    if decimal_sep:
        integer, fraction = cleaned.rsplit(decimal_sep, 1)
        integer = re.sub(r"[.,]", "", integer)
        if not re.fullmatch(r"\d{1,2}", fraction):
            return None
        candidate = f"{integer}.{fraction}"
    else:
        candidate = re.sub(r"[.,]", "", cleaned)
    try:
        return Decimal(candidate).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def find_money(text: str) -> list[Decimal]:
    cleaned = fix_digit_confusions(strip_accents(text)).casefold()
    found: list[Decimal] = []
    for match in MONEY.finditer(cleaned):
        raw = match.group(0)
        has_prefix = raw.startswith("r$")
        if not has_prefix and match.group(2) is None and "." not in match.group(1):
            continue  # número solto sem centavos e sem R$: provavelmente não é dinheiro
        if parsed := parse_money(raw):
            found.append(parsed)
    return found


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", fix_digit_confusions(text))


def infer_kind(field: str, value: str) -> str:
    name = normalize_text(field)
    if any(token in name for token in ("data", "date", "dt", "emissao", "entrega")):
        return "date"
    if any(token in name for token in ("valor", "value", "total", "preco", "amount", "montante")):
        return "money"
    if any(token in name for token in ("codigo", "code", "ref", "referencia", "chave", "rastreio",
                                       "tracking", "nf", "nota", "lote", "serie")):
        return "code"
    if any(token in name for token in ("pedido", "order", "numero", "id", "cpf", "cnpj", "telefone")):
        return "number"
    if parse_date(value):
        return "date"
    if value.strip().casefold().startswith("r$") or re.fullmatch(r"\d{1,3}(\.\d{3})*,\d{2}", value.strip()):
        return "money"
    if re.fullmatch(r"\d[\d.\-/ ]*", value.strip()):
        return "number"
    if re.fullmatch(r"[A-Za-z0-9\-./]+", value.strip()) and any(ch.isdigit() for ch in value):
        return "code"
    return "text"


@dataclass
class FieldMatch:
    field: str
    kind: str
    matched: bool
    score: float
    method: str
    found: str | None = None
    conflict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def match_field(field: str, value: str, text: str, fuzzy_threshold: int = 85, code_threshold: int = 90) -> FieldMatch:
    kind = infer_kind(field, value)
    value = value.strip()

    if kind == "date":
        expected = parse_date(value)
        if expected:
            candidates = find_dates(text)
            if expected in candidates:
                return FieldMatch(field, kind, True, 100.0, "date", expected.isoformat())
            found = candidates[0].isoformat() if candidates else None
            return FieldMatch(field, kind, False, 0.0, "date", found, conflict=bool(candidates))
        kind = "text"

    if kind == "money":
        expected = parse_money(value)
        if expected is not None:
            candidates = find_money(text)
            if expected in candidates:
                return FieldMatch(field, kind, True, 100.0, "money", str(expected))
            found = str(candidates[0]) if candidates else None
            return FieldMatch(field, kind, False, 0.0, "money", found, conflict=bool(candidates))
        kind = "text"

    if kind == "number":
        expected = digits_only(value)
        if expected:
            runs = DIGIT_RUN.findall(fix_digit_confusions(text))
            if any(expected == run or (len(expected) >= 4 and expected in run) for run in runs):
                return FieldMatch(field, kind, True, 100.0, "number", expected)
            best, best_run = 0.0, None
            for run in runs:
                score = fuzz.ratio(expected, run)
                if score > best:
                    best, best_run = score, run
            fuzzy_ok = len(expected) >= 6 and best >= code_threshold
            return FieldMatch(field, kind, fuzzy_ok, best, "number_fuzzy", best_run, conflict=bool(runs) and not fuzzy_ok)
        kind = "text"

    if kind == "code":
        expected = normalize_code(value)
        haystack = normalize_code(text)
        if expected and expected in haystack:
            return FieldMatch(field, kind, True, 100.0, "code", expected)
        score = fuzz.partial_ratio(expected, haystack) if expected and haystack else 0.0
        return FieldMatch(field, kind, score >= code_threshold, float(score), "code_fuzzy", None)

    expected = normalize_text(value)
    haystack = normalize_text(text)
    if not expected:
        return FieldMatch(field, "text", True, 100.0, "empty")
    if expected in haystack:
        return FieldMatch(field, "text", True, 100.0, "text", expected)
    partial = fuzz.partial_ratio(expected, haystack) if haystack else 0.0
    tokens = fuzz.token_set_ratio(expected, haystack) if haystack else 0.0
    score = max(partial, tokens)
    return FieldMatch(field, "text", score >= fuzzy_threshold, float(score), "text_fuzzy", None)
