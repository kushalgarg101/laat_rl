import pytest

from rl.laat_game.env import LaatCardEnv
from rl.laat_game.teachers import MixedTeacher, normalize_checkpoint_weights


def test_multi_teacher_weights_default_to_model4_mix():
    weights = normalize_checkpoint_weights(
        checkpoint_weight=None,
        checkpoint_count=2,
        heuristic_weight=None,
        random_weight=None,
    )

    assert weights == pytest.approx([0.45, 0.30])


def test_multi_teacher_normalizes_active_weights_and_returns_legal_actions(tmp_path, monkeypatch):
    class FakeModel:
        def predict(self, obs, action_masks, deterministic=True):
            return int(action_masks.nonzero()[0][0]), None

    def fake_load(path, device):
        return FakeModel()

    monkeypatch.setattr("rl.laat_game.teachers.MaskablePPO.load", fake_load)
    checkpoint_a = tmp_path / "model_a.zip"
    checkpoint_b = tmp_path / "model_b.zip"
    checkpoint_a.write_text("fake", encoding="utf-8")
    checkpoint_b.write_text("fake", encoding="utf-8")
    teacher = MixedTeacher([checkpoint_a, checkpoint_b], device="cpu", seed=1)
    env = LaatCardEnv(seed=1)
    env.reset(seed=1)

    assert sum(teacher.weights) == pytest.approx(1.0)
    assert set(teacher.teacher_id_map) >= {"heuristic", "random", "checkpoint_0", "checkpoint_1"}
    for _ in range(25):
        action, teacher_name = teacher.choose(env)
        assert env.action_masks()[action]
        assert teacher_name in teacher.teacher_id_map


def test_missing_teacher_checkpoint_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        MixedTeacher([tmp_path / "missing.zip"], device="cpu")
