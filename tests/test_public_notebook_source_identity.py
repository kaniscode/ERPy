"""The public notebook must reject changed sources before signal analysis."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from validation import validate_public_spes as public


def test_pinned_recipe_uses_explicit_versions_and_fixed_settings():
    path = Path(__file__).resolve().parents[1] / "validation/public_ds003708_source_manifest.json"
    manifest = json.loads(path.read_text())
    pinned = public._pin_dataset(public.DATASETS["ds003708"], manifest, manifest["settings"])
    for key, source in manifest["objects"].items():
        assert "?versionId=" in getattr(pinned, key)
        assert source["version_id"] in getattr(pinned, key)
    with pytest.raises(ValueError, match="settings mismatch"):
        public._pin_dataset(public.DATASETS["ds003708"], manifest, {**manifest["settings"], "max_events": 8})


def test_changed_event_bytes_fail_before_parsing(monkeypatch):
    payload = b"onset\telectrical_stimulation_site\n1\tA1-A2\n"
    monkeypatch.setattr(public.requests, "get", lambda *a, **k: SimpleNamespace(
        content=payload, text=payload.decode(), raise_for_status=lambda: None))
    expected = hashlib.sha256(payload).hexdigest()
    assert len(public.download_events(public.DATASETS["ds003708"], expected)) == 1
    with pytest.raises(ValueError, match="checksum mismatch"):
        public.download_events(public.DATASETS["ds003708"], "0" * 64)


@pytest.mark.parametrize("status,content_range,payload", [
    (200, "bytes 4-7/12", b"abcd"),
    (206, "bytes 0-3/12", b"abcd"),
    (206, "bytes 4-7/12", b"abc"),
])
def test_wrong_signal_range_is_rejected(monkeypatch, status, content_range, payload):
    monkeypatch.setattr(public.requests, "get", lambda *a, **k: SimpleNamespace(
        content=payload, status_code=status, headers={"Content-Range": content_range},
        raise_for_status=lambda: None))
    with pytest.raises((ValueError, RuntimeError)):
        public.fetch_range("https://example.org/signal?versionId=pinned", 4, 8)


def test_pinned_recipe_cannot_fall_back_to_summary_cache(tmp_path):
    path = Path(__file__).resolve().parents[1] / "validation/public_ds003708_source_manifest.json"
    with pytest.raises(ValueError, match="summary-cache reuse"):
        public.run_dataset(public.DATASETS["ds003708"], tmp_path, prefer_cache=True,
                           source_manifest=json.loads(path.read_text()))
