import torch
import pandas as pd
from pathlib import Path
from transformer_lens import HookedTransformer
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path("../results/activations").mkdir(parents=True, exist_ok=True)

print("Loading model...")
model = HookedTransformer.from_pretrained(
    "Qwen/Qwen-7B",
    device=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

print("Loading dataset...")
df = pd.read_csv("../results/final_dataset.csv")

n_layers = model.cfg.n_layers
layers_to_save = list(range(n_layers - 1, -1, -3))
print(f"Model has {n_layers} layers, saving: {layers_to_save}")

def extract_activations(prompt):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    last_token_idx = tokens.shape[1] - 1
    activations = {}
    
    for layer_idx in layers_to_save:
        acts = cache[f"blocks.{layer_idx}.hook_resid_post"]
        activations[f"layer_{layer_idx}"] = acts[0, last_token_idx, :].cpu().numpy()
    
    return activations

print(f"\nExtracting activations for {len(df)} examples...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    activations = extract_activations(row['full_prompt'])
    
    save_data = {
        'example_id': idx,
        'label': row['label'],
        'injection_type': row['injection_type'],
        'activations': activations
    }
    
    output_file = Path(f"../results/activations/example_{idx}.npz")
    np.savez_compressed(
        output_file,
        **{k: v for k, v in activations.items()},
        label=row['label'],
        injection_type=row['injection_type']
    )

print(f"✓ Saved activations to ../results/activations/")
print(f"  Total files: {len(list(Path('../results/activations').glob('*.npz')))}")
