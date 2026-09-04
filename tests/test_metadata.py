import io

import numpy as np
import pytest
from PIL import Image

from fraud_detector import AnalysisConfig, analyze_image
from fraud_detector.evaluation.synth import ReceiptSpec, jpeg_bytes, png_bytes, render_receipt, with_exif
from fraud_detector.jpegutil import STD_CHROMA, STD_LUMA, scaled_standard_table
from fraud_detector.signals.metadata import classify_software, parse_exif_datetime, read_exif


@pytest.fixture(scope="module")
def plain_jpeg():
    return jpeg_bytes(render_receipt(ReceiptSpec(seed=11)), 85)


def codes(result):
    return {item["code"]: item for item in result["findings"]}


def test_known_editor_scores_high(plain_jpeg):
    result = analyze_image(with_exif(plain_jpeg, software="Adobe Photoshop 25.0 (Windows)"), "editada.jpg")
    item = codes(result)["editing_software"]
    assert item["points"] == 25 and item["severity"] == "alto"
    assert result["features"]["metadata.software_kind"] == "editor"


@pytest.mark.parametrize("software", ["17.5.1", "S918BXXU3CWL1", "HDR+ 1.0.345", "Google", "NX3000-1.05"])
def test_firmware_strings_are_not_penalized(plain_jpeg, software):
    result = analyze_image(with_exif(plain_jpeg, software=software, make="Apple"), "foto.jpg")
    assert "editing_software" not in codes(result)
    assert "unknown_software" not in codes(result)
    assert result["features"]["metadata.software_kind"] == "firmware"
    assert result["features"]["metadata.has_camera_info"] is True


def test_unknown_software_is_informational(plain_jpeg):
    result = analyze_image(with_exif(plain_jpeg, software="MeuApp Fotos 2.0"), "foto.jpg")
    assert codes(result)["unknown_software"]["points"] == 6


def test_missing_exif_on_jpeg_is_weak_signal(plain_jpeg):
    result = analyze_image(plain_jpeg, "whatsapp.jpg")
    assert codes(result)["no_exif"]["points"] == 5
    png = analyze_image(png_bytes(render_receipt()), "captura.png")
    assert "no_exif" not in codes(png)


def test_capture_date_is_compared_with_expected_date(plain_jpeg):
    early = with_exif(plain_jpeg, datetime_original="2026:09:01 10:00:00")
    result = analyze_image(early, "foto.jpg", expected={"data": "04/09/2026"})
    item = codes(result)["exif_date_mismatch"]
    assert item["points"] == 20 and "3 dia" in item["evidence"]
    assert result["features"]["metadata.capture_gap_days"] == 3

    same_day = with_exif(plain_jpeg, datetime_original="2026:09:04 08:00:00")
    assert "exif_date_mismatch" not in codes(analyze_image(same_day, "foto.jpg", expected={"data": "2026-09-04"}))

    tolerant = AnalysisConfig(exif_date_tolerance_days=5)
    assert "exif_date_mismatch" not in codes(analyze_image(early, "foto.jpg", expected={"data": "04/09/2026"},
                                                           config=tolerant))


def test_capture_date_in_the_future_is_flagged(plain_jpeg):
    result = analyze_image(with_exif(plain_jpeg, datetime_original="2099:01:01 00:00:00"), "foto.jpg")
    assert codes(result)["exif_date_future"]["points"] == 10


def test_file_modified_long_after_capture(plain_jpeg):
    data = with_exif(plain_jpeg, datetime_original="2026:09:04 08:00:00", datetime_modified="2026:09:04 10:30:00")
    result = analyze_image(data, "foto.jpg")
    assert codes(result)["exif_modified_later"]["points"] == 10
    assert result["features"]["metadata.modified_after_capture_minutes"] == 150.0

    quick = with_exif(plain_jpeg, datetime_original="2026:09:04 08:00:00", datetime_modified="2026:09:04 08:02:00")
    assert "exif_modified_later" not in codes(analyze_image(quick, "foto.jpg"))


def test_nonstandard_quantization_tables_are_noticed(plain_jpeg):
    luma = scaled_standard_table(75, STD_LUMA).astype(int).tolist()
    chroma = scaled_standard_table(75, STD_CHROMA).astype(int).tolist()
    luma[5] += 7
    luma[20] += 9
    data = with_exif(plain_jpeg, qtables=[luma, chroma])
    result = analyze_image(data, "editor.jpg")
    assert result["signals"]["metadata"]["jpeg"]["standard_tables"] is False
    assert "nonstandard_quantization" in codes(result)
    assert result["features"]["metadata.jpeg_standard_tables"] is False
    assert "nonstandard_quantization" not in codes(analyze_image(plain_jpeg, "normal.jpg"))


def test_exif_reader_includes_sub_ifd_tags(plain_jpeg):
    data = with_exif(plain_jpeg, software="X", datetime_original="2026:09:04 08:00:00", make="Samsung")
    exif = read_exif(Image.open(io.BytesIO(data)))
    assert exif["DateTimeOriginal"] == "2026:09:04 08:00:00"
    assert exif["Make"] == "Samsung"


def test_helpers():
    config = AnalysisConfig()
    assert classify_software("", config) == "none"
    assert classify_software("GIMP 2.10", config) == "editor"
    assert classify_software("17.5.1", config) == "firmware"
    assert parse_exif_datetime("2026:09:04 08:00:00").hour == 8
    assert parse_exif_datetime("lixo") is None
    assert parse_exif_datetime(None) is None
