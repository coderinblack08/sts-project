import numpy as np
import torch
import torch.nn.functional as F


def get_last_attn(attn_map):
    for i, layer in enumerate(attn_map):
        attn_map[i] = layer[:, :, -1, :].unsqueeze(2)

    return attn_map


def sample_token(logits, top_k=None, top_p=None, temperature=1.0):
    # Optionally apply temperature
    logits = logits / temperature

    # Apply top-k sampling
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))  # Ensure top_k <= vocab size
        values, indices = torch.topk(logits, top_k)
        probs = F.softmax(values, dim=-1)
        next_token_id = indices[torch.multinomial(probs, 1)]

        return next_token_id

    return logits.argmax(dim=-1).squeeze()


def process_attn(attention, rng, attn_func):
    heatmap = np.zeros((len(attention), attention[0].shape[1]))
    for i, attn_layer in enumerate(attention):
        attn_layer = attn_layer.to(torch.float32).numpy()

        if "sum" in attn_func:
            last_token_attn_to_inst = np.sum(attn_layer[0, :, -1, rng[0][0] : rng[0][1]], axis=1)
            attn = last_token_attn_to_inst

        elif "max" in attn_func:
            last_token_attn_to_inst = np.max(attn_layer[0, :, -1, rng[0][0] : rng[0][1]], axis=1)
            attn = last_token_attn_to_inst

        else:
            raise NotImplementedError

        last_token_attn_to_inst_sum = np.sum(attn_layer[0, :, -1, rng[0][0] : rng[0][1]], axis=1)
        last_token_attn_to_data_sum = np.sum(attn_layer[0, :, -1, rng[1][0] : rng[1][1]], axis=1)

        if "normalize" in attn_func:
            epsilon = 1e-8
            heatmap[i, :] = attn / (
                last_token_attn_to_inst_sum + last_token_attn_to_data_sum + epsilon
            )
        else:
            heatmap[i, :] = attn

    heatmap = np.nan_to_num(heatmap, nan=0.0)

    return heatmap


def calc_attn_score(heatmap, heads):
    score = np.mean([heatmap[l, h] for l, h in heads], axis=0)
    return score
