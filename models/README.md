# Probe

```bash
cd sts-project/models/drift-tracker
uv run activations.py --n max --split train --hf --hf-repo [repo name] &
uv run activations.py --n max --split test --hf --hf-repo [repo name] &
wait
uv run t-sne.py --hf --hf-repo [repo name] --file train_activations_max.pt
uv run regression-probe.py --hf --hf-repo [repo name] --train_file train_activations_max.pt --test_file test_activations_max.pt
uv run mass-mean-probe.py --hf --hf-repo [repo name] --train_file train_activations_max.pt --test_file test_activations_max.pt
```
