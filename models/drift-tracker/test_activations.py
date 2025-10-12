import torch
from pathlib import Path
import subprocess

output_dir = Path(__file__).parent / "output"


def test_single_sample():
    subprocess.run(
        ["uv", "run", "python", "drift-tracker/activations.py", "--n", "1"],
        cwd=Path(__file__).parent.parent,
        check=True,
    )

    data = torch.load(output_dir / "activations_1.pt")

    assert "samples" in data
    assert "global_metadata" in data
    assert len(data["samples"]) == 1
    assert data["global_metadata"]["num_samples"] == 1


def test_multiple_samples():
    subprocess.run(
        ["uv", "run", "python", "drift-tracker/activations.py", "--n", "5"],
        cwd=Path(__file__).parent.parent,
        check=True,
    )

    data = torch.load(output_dir / "activations_5.pt")

    assert len(data["samples"]) == 5
    assert data["global_metadata"]["num_samples"] == 5

    for idx, sample in enumerate(data["samples"]):
        assert "clean" in sample
        assert "poisoned" in sample
        assert "metadata" in sample
        assert sample["metadata"]["sample_idx"] == idx

        required_metadata = [
            "sample_idx",
            "injection_start_token",
            "injection_end_token",
            "clean_last_token",
            "poisoned_last_token",
        ]
        for field in required_metadata:
            assert field in sample["metadata"]

        assert len(sample["metadata"]) == 5

        num_layers = data["global_metadata"]["num_layers"]
        d_model = data["global_metadata"]["d_model"]

        for pos_key, activations in sample["clean"].items():
            assert activations.shape == (num_layers, d_model)

        for pos_key, activations in sample["poisoned"].items():
            assert activations.shape == (num_layers, d_model)
