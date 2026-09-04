import hashlib

from fraud_detector import analyze_image
from fraud_detector.evaluation.synth import (
    ReceiptSpec, forge_value_edit, jpeg_bytes, open_rgb, render_receipt, screenshot_like, whatsapp_like,
)
from fraud_detector.hashing import BKTree, HashStore, ListHashIndex, as_index, dhash, hamming_distance, phash


def test_phash_survives_whatsapp_and_screenshot_but_separates_receipts():
    original = render_receipt(ReceiptSpec(seed=1))
    data = jpeg_bytes(original, 90)
    p0, d0 = phash(original), dhash(original)

    resent = open_rgb(whatsapp_like(data))
    assert len(p0) == 64  # 256 bits
    assert hamming_distance(p0, phash(resent)) <= 8
    assert hamming_distance(d0, dhash(resent)) <= 6

    # Mesmo modelo de comprovante, outro texto: o hash de 64 bits confundiria (só vê o layout).
    for seed in (8, 9, 10):
        other = render_receipt(ReceiptSpec(order_id="99999", recipient="OUTRA PESSOA", value="R$ 77,00", seed=seed))
        assert hamming_distance(p0, phash(other)) > 20


def test_bktree_radius_search():
    tree = BKTree()
    base = int("0f0f0f0f0f0f0f0f", 16)
    for shift in (0, 1, 2, 5, 20):
        value = base ^ ((1 << shift) - 1)  # vira `shift` bits
        tree.add(f"{value:016x}", f"d{shift}")
    hits = tree.search("0f0f0f0f0f0f0f0f", 2)
    assert [payload for _, payload in hits] == ["d0", "d1", "d2"]
    assert tree.search("0f0f0f0f0f0f0f0f", 0)[0][1] == "d0"
    assert BKTree().search("0f0f0f0f0f0f0f0f", 5) == []


def test_store_persists_and_finds_exact_and_near(tmp_path):
    original = render_receipt(ReceiptSpec(seed=2))
    data = jpeg_bytes(original, 85)
    sha = hashlib.sha256(data).hexdigest()
    path = tmp_path / "hashes.sqlite"

    store = HashStore(path)
    store.add(sha, dhash(original), phash(original), "pedido-1.jpg", reference="PED-1")
    assert store.count() == 1
    store.close()

    reopened = HashStore(path)
    assert reopened.find_exact(sha)[0]["reference"] == "PED-1"
    resent = open_rgb(whatsapp_like(data))
    near = reopened.find_near(dhash(resent), phash(resent), 7, 10)
    assert near and near["within"] and near["match"]["reference"] == "PED-1"

    other = render_receipt(ReceiptSpec(order_id="55555", seed=8))
    far = reopened.find_near(dhash(other), phash(other), 7, 10)
    assert far is None or not far["within"]
    reopened.close()


def test_signal_reports_exact_and_near_duplicates():
    original = render_receipt(ReceiptSpec(seed=3))
    data = jpeg_bytes(original, 85)
    store = HashStore()
    first = analyze_image(data, "primeiro.jpg", known_hashes=store)
    assert not [f for f in first["findings"] if f["code"].endswith("duplicate")]
    store.add(first["sha256"], first["perceptual_hash"], first["features"]["duplicates.phash"], "primeiro.jpg", "PED-1")

    again = analyze_image(data, "reenvio.jpg", known_hashes=store)
    codes = {f["code"]: f for f in again["findings"]}
    assert codes["exact_duplicate"]["points"] == 45 and "PED-1" in codes["exact_duplicate"]["evidence"]
    assert "near_duplicate" not in codes  # o exato já cobre o caso
    assert again["features"]["duplicates.exact_duplicate"] is True

    resent = analyze_image(whatsapp_like(data), "whatsapp.jpg", known_hashes=store)
    codes = {f["code"]: f for f in resent["findings"]}
    assert "exact_duplicate" not in codes and codes["near_duplicate"]["points"] == 35
    assert resent["nearest_duplicate"]["reference"] == "PED-1"

    forged, _ = forge_value_edit(data)
    edited = analyze_image(forged, "editada.jpg", known_hashes=store)
    assert any(f["code"] == "near_duplicate" for f in edited["findings"])


def test_legacy_list_index_still_works():
    original = render_receipt(ReceiptSpec(seed=4))
    data = jpeg_bytes(original, 85)
    result = analyze_image(data, "a.jpg", known_hashes=[dhash(original)])
    assert any(f["code"] == "near_duplicate" for f in result["findings"])
    assert result["nearest_duplicate"]["distance"] <= 2  # o JPEG decodificado difere da renderização em poucos bits
    assert isinstance(as_index(["0" * 16]), ListHashIndex)
    assert as_index(None) is None
    assert analyze_image(data, "b.jpg", known_hashes=[])["nearest_duplicate"] is None
