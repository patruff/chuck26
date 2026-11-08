import hashlib
import sys
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import generate_film_pairs as batch
from run_gguf_test import pod_config


def test_paired_slots_give_every_request_the_original_context():
    config = pod_config("test", "fixture", "NVIDIA L40S", "SECURE", 1536, 16384, 6)
    assert "-c 98304 -np 6" in config["dockerStartCmd"][0]


def test_resume_keeps_successful_samples_and_retries_only_the_missing_side(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(batch, "worker", {}, raising=False)
    generate = Mock(
        side_effect=[
            {"state": "success", "parody": "A"},
            {"error": "temporary"},
            {"error": "temporary"},
            {"state": "success", "parody": "B"},
        ]
    )
    monkeypatch.setattr(batch, "generate_title", generate)
    film = {"rank": "1", "title": "The Matrix", "year": "1999"}
    a = film, "a", tmp_path, "signature"
    b = film, "b", tmp_path, "signature"
    assert batch.generate_side(a) and not batch.generate_side(b)
    original = (tmp_path / "001-a.json.gz").read_bytes()
    assert batch.generate_side(a) and batch.generate_side(b)
    assert original == (tmp_path / "001-a.json.gz").read_bytes()
    assert len(batch.read_gzip(tmp_path / "001-b.json.gz")["failed_attempts"]) == 2
    assert [v.args[1] for v in generate.call_args_list] == [1042, 1542, 1543, 1544]
