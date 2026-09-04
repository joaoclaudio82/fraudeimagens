import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from fraud_detector import AnalysisConfig, analyze_image
from fraud_detector.evaluation.synth import ReceiptSpec, forge_copy_move, jpeg_bytes, render_receipt
from fraud_detector.signals.copy_move import detect_copy_move


def inside(box, target):
    """Centro da caixa detectada dentro da região declarada (o texto é menor que a caixa da forjaria)."""
    cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
    return target["x"] <= cx <= target["x"] + target["w"] and target["y"] <= cy <= target["y"] + target["h"]


def copy_move_findings(result):
    return [item for item in result["findings"] if item["code"] == "copy_move"]


@pytest.fixture(scope="module")
def receipt():
    return render_receipt(ReceiptSpec(seed=21))


def test_pasted_block_is_detected_with_both_boxes(receipt):
    forged, source, target = forge_copy_move(jpeg_bytes(receipt, 85), quality=88)
    result = analyze_image(forged, "copy_move.jpg")
    found = copy_move_findings(result)
    assert found, result["signals"]["copy_move"]
    clusters = result["signals"]["copy_move"]["clusters"]
    assert len(clusters) == 1, clusters  # a cópia é reportada uma vez, não nos dois sentidos
    boxes = [found[0]["region"], clusters[0]["source"], clusters[0]["target"]]
    assert any(inside(box, target) for box in boxes) and any(inside(box, source) for box in boxes)
    assert clusters[0]["occurrences"] == 1 and clusters[0]["cell_agreement"] >= 0.8
    assert result["features"]["copy_move.best_correlation"] >= 0.85
    assert result["_images"]["copy_move.overlay"].size == receipt.size


def test_intact_receipt_has_no_copy_move(receipt):
    result = analyze_image(jpeg_bytes(receipt, 85), "integra.jpg")
    assert copy_move_findings(result) == []
    assert result["features"]["copy_move.copy_move_found"] is False


def test_repeated_labels_in_a_table_are_not_copy_move():
    """'TOTAL R$' repetido em várias linhas gera casamentos ORB, mas os recortes inteiros não se correlacionam."""
    image = Image.new("RGB", (1000, 800), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=26)
    rng = np.random.default_rng(0)
    for row in range(10):
        amount = f"{rng.integers(10, 999)},{rng.integers(0, 99):02d}"
        draw.text((80, 60 + row * 60), f"ITEM {row + 1:02d}  TOTAL R$ {amount}", fill=(20, 20, 20), font=font)
    noisy = np.clip(np.asarray(image).astype(np.int16) + rng.normal(0, 3, (800, 1000, 3)), 0, 255).astype(np.uint8)
    result = analyze_image(jpeg_bytes(Image.fromarray(noisy), 85), "tabela.jpg")
    assert copy_move_findings(result) == [], result["signals"]["copy_move"]["clusters"]


def test_detector_can_be_disabled(receipt):
    result = analyze_image(jpeg_bytes(receipt, 85), "integra.jpg", config=AnalysisConfig(copy_move_enabled=False))
    assert result["features"]["copy_move.copy_move_found"] is False
    assert "copy_move.keypoints" not in result["features"]


def test_flat_image_has_no_keypoints():
    analysis = detect_copy_move(np.full((300, 300), 128, dtype=np.uint8), AnalysisConfig())
    assert analysis.keypoints < 20 and analysis.confirmed == []
