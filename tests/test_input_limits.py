import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fraud_detector import AnalysisConfig, analyze_image
from fraud_detector.api import create_app, config_from_env
from fraud_detector.context import ImageContext, ImageLimitError
from fraud_detector.storage import AnalysisStore


def png():
    buffer = io.BytesIO()
    Image.new('RGB', (20, 10), 'white').save(buffer, 'PNG')
    return buffer.getvalue()


def test_limits_apply_before_decode_and_accept_exact_boundaries(monkeypatch):
    data = png()
    config = AnalysisConfig(max_upload_bytes=len(data), max_image_pixels=200)
    assert ImageContext.from_bytes(data, config=config).width == 20
    with pytest.raises(ImageLimitError, match='bytes'):
        analyze_image(data, config=config.with_overrides(max_upload_bytes=len(data) - 1))
    def must_not_decode(*args, **kwargs):
        pytest.fail('pixel limit must be checked before image.load')
    monkeypatch.setattr('PIL.PngImagePlugin.PngImageFile.load', must_not_decode)
    with pytest.raises(ImageLimitError, match='pixels'):
        analyze_image(data, config=config.with_overrides(max_image_pixels=199))


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '20'])
@pytest.mark.parametrize('name', ['max_upload_bytes', 'max_image_pixels'])
def test_invalid_limits(name, value):
    with pytest.raises(ValueError):
        AnalysisConfig(**{name: value})


def test_environment_limits(monkeypatch):
    monkeypatch.setenv('IMAGEGUARD_MAX_UPLOAD_BYTES', '1024')
    monkeypatch.setenv('IMAGEGUARD_MAX_IMAGE_PIXELS', '200')
    config = config_from_env()
    assert config.max_upload_bytes == 1024 and config.max_image_pixels == 200


@pytest.mark.parametrize('config', [AnalysisConfig(max_upload_bytes=5), AnalysisConfig(max_image_pixels=199)])
def test_api_rejects_oversized_input_without_persisting(config):
    store = AnalysisStore()
    with TestClient(create_app(store, config)) as client:
        response = client.post('/analyze', files={'file': ('x.png', png())})
        assert response.status_code == 413
        assert store.stats()['analyses'] == 0
        assert store.stats()['hashes'] == 0
    store.close()


@pytest.mark.parametrize('expected', ['[]', 'null', 'true', '{"valor": []}', '{"valor": true}',
                                      '{"valor": NaN}', '{"valor": Infinity}', '{"outro": "x"}'])
def test_api_rejects_invalid_expected(expected):
    with TestClient(create_app(AnalysisStore(), AnalysisConfig())) as client:
        response = client.post('/analyze', files={'file': ('x.png', png())}, data={'expected': expected})
        assert response.status_code == 400


def test_expected_zero_and_explicit_field_override(monkeypatch):
    captured = []
    def analyze(data, filename, fields, *args):
        captured.append(fields)
        return {}
    monkeypatch.setattr('fraud_detector.api.analyze_image', analyze)
    with TestClient(create_app(AnalysisStore(), AnalysisConfig())) as client:
        for extra in ({}, {'pedido': 'novo'}):
            response = client.post('/analyze', files={'file': ('x.png', png())},
                                   data={'expected': '{"valor": 0, "pedido": "antigo"}', 'persist': 'false', **extra})
            assert response.status_code == 200
    assert captured == [{'valor': '0', 'pedido': 'antigo'}, {'valor': '0', 'pedido': 'novo'}]


@pytest.mark.parametrize('path', ['/analyses', '/review-queue'])
@pytest.mark.parametrize('limit', [0, -1, 501])
def test_invalid_page_limits(path, limit):
    with TestClient(create_app(AnalysisStore(), AnalysisConfig())) as client:
        assert client.get(path, params={'limit': limit}).status_code == 422


def test_invalid_decision_filters():
    with TestClient(create_app(AnalysisStore(), AnalysisConfig())) as client:
        assert client.get('/analyses', params={'decision': 'typo'}).status_code == 422
        assert client.get('/review-queue', params={'min_decision': 'typo'}).status_code == 422


@pytest.fixture(autouse=True)
def local_development_auth(monkeypatch):
    monkeypatch.setenv("IMAGEGUARD_AUTH_MODE", "disabled")
