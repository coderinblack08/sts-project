uv run activations.py --n max --split train --hf --hf-repo kevintlu/qwen-2.5-activations &
uv run activations.py --n max --split test --hf --hf-repo kevintlu/qwen-2.5-activations &
wait
uv run t-sne.py --hf --hf-repo kevintlu/qwen-2.5-activations --file train_activations_max.pt
uv run regression-probe.py --hf --hf-repo kevintlu/qwen-2.5-activations --train_file train_activations_max.pt --test_file test_activations_max.pt
uv run mass-mean-probe.py --hf --hf-repo kevintlu/qwen-2.5-activations --train_file train_activations_max.pt --test_file test_activations_max.pt