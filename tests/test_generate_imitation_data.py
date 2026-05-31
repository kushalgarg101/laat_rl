import sys

import numpy as np

from rl.scripts.generate_imitation_data import main


def test_generate_imitation_data_stores_teacher_names(tmp_path, monkeypatch):
    output = tmp_path / "tiny.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_imitation_data",
            "--decisions",
            "20",
            "--output",
            str(output),
            "--device",
            "cpu",
            "--opponent-pool",
            "random",
        ],
    )

    main()

    data = np.load(output)
    assert data["actions"].shape == (20,)
    assert data["teacher_ids"].shape == (20,)
    assert "teacher_names" in data.files
    assert set(data["teacher_names"].tolist()) >= {"heuristic", "random"}
