import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from fraud_detector.api import create_app
from fraud_detector.storage import AnalysisStore

TOKENS = {role: f'test-only-{role}-credential' for role in ('operator', 'reviewer', 'admin')}


def registry():
    return [{'subject': role + '-user', 'role': role,
             'token_sha256': hashlib.sha256(token.encode()).hexdigest()} for role, token in TOKENS.items()]


@pytest.fixture
def secured(tmp_path, monkeypatch):
    path = tmp_path / 'credentials.json'
    path.write_text(json.dumps(registry()))
    monkeypatch.setenv('IMAGEGUARD_AUTH_MODE', 'required')
    monkeypatch.setenv('IMAGEGUARD_AUTH_FILE', str(path))
    store = AnalysisStore()
    aid = store.save({'risk_score': 60, 'decision': 'REVISAR'}, register_hashes=False)
    with TestClient(create_app(store)) as client:
        yield client, store, aid, path
    store.close()


def headers(role):
    return {'Authorization': 'Bearer ' + TOKENS[role]}


@pytest.mark.parametrize('method,path,kwargs,allowed', [
    ('get', '/analyses', {}, {'operator', 'reviewer', 'admin'}),
    ('get', '/analyses/1', {}, {'operator', 'reviewer', 'admin'}),
    ('get', '/analyses/1/reviews', {}, {'reviewer', 'admin'}),
    ('get', '/review-queue', {}, {'reviewer', 'admin'}),
    ('get', '/stats', {}, {'reviewer', 'admin'}),
    ('get', '/export/labels', {}, {'admin'}),
    ('post', '/maintenance/purge', {}, {'admin'}),
    ('post', '/analyses/1/review', {'json': {'status': 'legitimate'}}, {'reviewer', 'admin'}),
    ('post', '/analyze', {'files': {'file': ('x.png', b'x')}, 'data': {'persist': 'false'}}, {'operator', 'admin'}),
])
def test_route_permission_matrix(secured, monkeypatch, method, path, kwargs, allowed):
    client, store, _, _ = secured
    monkeypatch.setattr('fraud_detector.api.analyze_image', lambda *args: {})
    for role in TOKENS:
        response = client.request(method, path, headers=headers(role), **kwargs)
        assert response.status_code == (200 if role in allowed else 403), response.text
    response = client.request(method, path, **kwargs)
    assert response.status_code == 401
    assert response.headers['www-authenticate'] == 'Bearer'


def test_identity_cannot_be_spoofed_and_legacy_events_are_unverified(secured):
    client, store, aid, _ = secured
    store.review(aid, 'pending', 'legacy-name')
    response = client.post(f'/analyses/{aid}/review', headers=headers('reviewer'),
                           json={'status': 'legitimate', 'reviewer': 'admin-user', 'expected_version': 1})
    assert response.status_code == 200
    events = store.review_history(aid)
    assert events[0]['reviewer_authenticated'] == 0
    assert events[1]['reviewer'] == 'reviewer-user'
    assert events[1]['reviewer_authenticated'] == 1
    assert store.get(aid)['reviewer'] == 'reviewer-user'


def test_revocation_and_invalid_configuration_fail_closed(secured):
    client, _, _, path = secured
    path.write_text(json.dumps([row for row in registry() if row['role'] != 'operator']))
    assert client.get('/analyses', headers=headers('operator')).status_code == 401
    for contents in ('not-json', '{}', '[]', '[{"role":"root"}]', json.dumps(registry() + registry())):
        path.write_text(contents)
        response = client.get('/analyses', headers=headers('admin'))
        assert response.status_code == 503
        assert str(path) not in response.text
    path.unlink()
    assert client.get('/analyses', headers=headers('admin')).status_code == 503


def test_default_requires_configuration_and_health_is_minimal(monkeypatch):
    monkeypatch.delenv('IMAGEGUARD_AUTH_MODE', raising=False)
    monkeypatch.delenv('IMAGEGUARD_AUTH_FILE', raising=False)
    with TestClient(create_app()) as client:
        assert client.get('/analyses').status_code == 503
        body = client.get('/health').json()
        assert set(body) == {'status', 'analysis_version'}
        assert body['status'] == 'ok'
    monkeypatch.setenv('IMAGEGUARD_AUTH_MODE', 'typo')
    with TestClient(create_app()) as client:
        assert client.get('/analyses').status_code == 503


@pytest.mark.parametrize('authorization', ['Bearer wrong', 'Basic abc', 'Bearer', ''])
def test_invalid_credentials(secured, authorization):
    client, _, _, _ = secured
    response = client.get('/analyses', headers={'Authorization': authorization})
    assert response.status_code == 401


def test_direct_database_ui_blocked_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv('IMAGEGUARD_AUTH_MODE', raising=False)
    db = tmp_path / 'must-not-open.sqlite'
    monkeypatch.setenv('IMAGEGUARD_DB', str(db))
    at = AppTest.from_file('app.py').run()
    assert not at.exception
    assert any('API' in item.value for item in at.error)
    assert not db.exists()


def test_denied_writes_do_not_change_or_delete_data(secured):
    client, store, aid, _ = secured
    store._conn.execute("UPDATE analyses SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?", (aid,))
    store._conn.commit()
    for role in ('operator', 'reviewer'):
        assert client.post('/maintenance/purge', headers=headers(role)).status_code == 403
        assert store.get(aid) is not None
    assert client.post(f'/analyses/{aid}/review', headers=headers('operator'),
                       json={'status': 'confirmed_fraud'}).status_code == 403
    assert store.review_history(aid) == []
    assert store.get(aid)['status'] == 'pending'
    assert client.post('/maintenance/purge', headers=headers('admin')).json() == {'removed': 1}
    assert store.get(aid) is None
