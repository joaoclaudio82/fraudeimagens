import io

from PIL import Image, ImageDraw, ImageFilter

from fraud_detector.analyzer import analyze_image, hamming_distance, perceptual_hash


def image_bytes(blur=False):
    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 90, 820, 510), outline="black", width=4)
    draw.text((120, 150), "PEDIDO 12345", fill="black")
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(8))
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def test_hash_is_stable():
    data = image_bytes()
    image = Image.open(io.BytesIO(data))
    assert perceptual_hash(image) == perceptual_hash(image.copy())
    assert hamming_distance(perceptual_hash(image), perceptual_hash(image)) == 0


def test_analysis_has_auditable_output():
    result = analyze_image(image_bytes(), "proof.png", expected={"pedido": "12345"})
    assert len(result["sha256"]) == 64
    assert 0 <= result["risk_score"] <= 100
    assert result["decision"] in {"BAIXO RISCO", "ATENÇÃO", "REVISAR"}
    assert result["disclaimer"]
    assert result.pop("_ela_image").size == (900, 600)


def test_blurred_image_is_flagged():
    result = analyze_image(image_bytes(blur=True), "blurred.png")
    assert any(item["code"] == "blur" for item in result["findings"])

