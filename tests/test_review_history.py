import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from fraud_detector.api import create_app
from fraud_detector.storage import AnalysisStore, ReviewConflict


def add(store):
    return store.save({'risk_score': 60, 'decision': 'REVISAR', 'filename': 'a.jpg'}, register_hashes=False)


def test_revisions_reopening_and_current_labels():
    store = AnalysisStore()
    aid = add(store)
    assert store.review_history(aid) == []
    assert store.get(aid)['review_version'] == 0
    store.review(aid, 'confirmed_fraud', 'ana', 'primeiro parecer', expected_version=0)
    store.review(aid, 'legitimate', 'bia', 'comprovante conferido', expected_version=1)
    assert [r['status'] for r in store.review_history(aid)] == ['confirmed_fraud', 'legitimate']
    assert store.labeled_rows()[0]['label'] == 0
    store.review(aid, 'pending', 'bia', 'reabrir', expected_version=2)
    assert store.labeled_rows() == []
    assert store.queue()[0]['review_version'] == 3
    assert store.list()[0]['review_version'] == 3
    assert store.review_history(aid, limit=1, after_version=1)[0]['reviewer'] == 'bia'
    assert store.review_history(aid)[0]['note'] == 'primeiro parecer'
    store.close()


def test_conflict_and_missing_analysis_leave_no_events():
    store = AnalysisStore()
    aid = add(store)
    store.review(aid, 'legitimate', expected_version=0)
    with pytest.raises(ReviewConflict):
        store.review(aid, 'confirmed_fraud', expected_version=0)
    with pytest.raises(KeyError):
        store.review(999, 'legitimate')
    with pytest.raises(KeyError):
        store.review_history(999)
    assert len(store.review_history(aid)) == 1
    assert store.get(aid)['status'] == 'legitimate'
    store.close()


def test_history_failure_rolls_back_current_review():
    store = AnalysisStore()
    aid = add(store)
    store._conn.execute("CREATE TRIGGER fail_history BEFORE INSERT ON review_history BEGIN SELECT RAISE(ABORT, 'test'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.review(aid, 'legitimate')
    assert store.get(aid)['status'] == 'pending'
    assert store.get(aid)['review_version'] == 0
    assert store.review_history(aid) == []
    store.close()


def test_migration_preserves_only_known_legacy_snapshot(tmp_path):
    path = tmp_path / 'legacy.sqlite'
    store = AnalysisStore(path)
    aid = add(store)
    store.close()
    with sqlite3.connect(path) as conn:
        conn.executescript('DROP TABLE review_history; DROP TABLE reviews; CREATE TABLE reviews '
                           '(analysis_id INTEGER PRIMARY KEY, status TEXT, reviewer TEXT, note TEXT, reviewed_at TEXT);')
        conn.execute('INSERT INTO reviews VALUES (?, ?, ?, ?, ?)',
                     (aid, 'legitimate', 'ana', 'legado', '2026-09-01T10:00:00+00:00'))
    for _ in range(2):
        store = AnalysisStore(path)
        events = store.review_history(aid)
        assert len(events) == 1 and events[0]['source'] == 'legacy_snapshot'
        assert events[0]['reviewed_at'] == '2026-09-01T10:00:00+00:00'
        assert store.get(aid)['review_version'] == 1
        store.close()
    store = AnalysisStore(path)
    store.review(aid, 'inconclusive', expected_version=1)
    store.close()
    store = AnalysisStore(path)
    assert [r['source'] for r in store.review_history(aid)] == ['legacy_snapshot', 'review']
    store.close()


@pytest.mark.parametrize('shared', [True, False])
def test_concurrent_reviews_have_one_winner(tmp_path, shared):
    first = AnalysisStore(tmp_path / 'shared.sqlite')
    aid = add(first)
    second = first if shared else AnalysisStore(tmp_path / 'shared.sqlite')
    barrier = Barrier(2)
    def review(store, status):
        barrier.wait(timeout=5)
        try:
            store.review(aid, status, expected_version=0)
            return 'saved'
        except ReviewConflict:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(review, first, 'legitimate'), pool.submit(review, second, 'confirmed_fraud')]
        assert sorted(f.result(timeout=10) for f in futures) == ['conflict', 'saved']
    assert len(first.review_history(aid)) == 1
    if not shared:
        second.close()
    first.close()


def test_expiry_purges_events_and_keeps_unexpired(tmp_path):
    store = AnalysisStore(tmp_path / 'retention.sqlite', retention_days=1)
    expired = add(store)
    store.review(expired, 'legitimate')
    store.retention_days = 30
    live = add(store)
    store.review(live, 'inconclusive')
    assert store.purge_expired(datetime.now(timezone.utc) + timedelta(days=2)) == 1
    assert store._conn.execute('SELECT COUNT(*) FROM review_history WHERE analysis_id=?', (expired,)).fetchone()[0] == 0
    assert len(store.review_history(live)) == 1
    store.close()


def test_api_history_conflict_pagination_and_legacy_client():
    store = AnalysisStore()
    aid = add(store)
    with TestClient(create_app(store)) as client:
        path = f'/analyses/{aid}'
        assert client.get(path + '/reviews').json() == []
        assert client.post(path + '/review', json={'status': 'legitimate', 'expected_version': 0}).status_code == 200
        assert client.post(path + '/review', json={'status': 'confirmed_fraud', 'expected_version': 0}).status_code == 409
        assert client.post(path + '/review', json={'status': 'pending'}).status_code == 200
        assert client.get(path).json()['review_version'] == 2
        assert client.get(path + '/reviews', params={'after_version': 1, 'limit': 1}).json()[0]['version'] == 2
        assert client.get('/analyses/999/reviews').status_code == 404
        for params in ({'limit': 0}, {'limit': 501}, {'after_version': -1}):
            assert client.get(path + '/reviews', params=params).status_code == 422
        for version in (-1, True, 0.5, '1'):
            assert client.post(path + '/review', json={'status': 'pending', 'expected_version': version}).status_code == 422
    store.close()


@pytest.fixture(autouse=True)
def local_development_auth(monkeypatch):
    monkeypatch.setenv("IMAGEGUARD_AUTH_MODE", "disabled")
