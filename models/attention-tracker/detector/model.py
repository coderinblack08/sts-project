import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_last_attn, sample_token

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available() else 'cpu'
)


class Model:
    def __init__(self, config):
        self.provider = config["model_info"]["provider"]
        self.name = config["model_info"]["name"]
        self.temperature = float(config["params"]["temperature"])

    def print_model_info(self):
        print(
            f"{'-'*len(f'| Model name: {self.name}')}\n| Provider: {self.provider}\n| Model name: {self.name}\n{'-'*len(f'| Model name: {self.name}')}"
        )

    def set_API_key(self):
        raise NotImplementedError(
            "ERROR: Interface doesn't have the implementation for set_API_key"
        )

    def query(self):
        raise NotImplementedError(
            "ERROR: Interface doesn't have the implementation for query"
        )


class AttentionModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.name = config["model_info"]["name"]
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        model_id = config["model_info"]["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",
        ).eval()

        self.top_k = 50
        self.top_p = None

        if config["params"].get("important_heads", "all") == "all":
            attn_size = self.get_map_dim()
            self.important_heads = [
                [i, j] for i in range(attn_size[0]) for j in range(attn_size[1])
            ]
        else:
            self.important_heads = config["params"]["important_heads"]

    def get_map_dim(self):
        _, _, attention_maps, _, _, _ = self.inference("print hi", "")
        attention_map = attention_maps[0]
        return len(attention_map), attention_map[0].shape[1]

    def inference(self, instruction, data, max_output_tokens=None):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": "Data: " + data},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        instruction_len = len(self.tokenizer.encode(instruction))
        data_len = len(self.tokenizer.encode(data))

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_tokens = self.tokenizer.convert_ids_to_tokens(
            model_inputs['input_ids'][0]
        )

        if "qwen" in self.name:
            data_range = ((3, 3 + instruction_len), (-5 - data_len, -5))
        elif "phi3" in self.name:
            data_range = ((1, 1 + instruction_len), (-2 - data_len, -2))
        elif "llama3-8b" in self.name:
            data_range = ((5, 5 + instruction_len), (-5 - data_len, -5))
        elif "mistral-7b" in self.name:
            data_range = ((3, 3 + instruction_len), (-1 - data_len, -1))
        elif "granite3-8b" in self.name:
            data_range = ((3, 3 + instruction_len), (-5 - data_len, -5))
        elif "gemma2-9b" in self.name:
            data_range = ((3, 3 + instruction_len), (-5 - data_len, -5))
        else:
            raise NotImplementedError

        generated_tokens = []
        generated_probs = []
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask

        attention_maps = []

        if max_output_tokens != None:
            n_tokens = max_output_tokens
        else:
            n_tokens = self.max_output_tokens

        with torch.no_grad():
            for i in range(n_tokens):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )

                logits = output.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token_id = sample_token(
                    logits[0], top_k=self.top_k, top_p=self.top_p, temperature=1.0
                )[0]

                generated_probs.append(probs[0, next_token_id.item()].item())
                generated_tokens.append(next_token_id.item())

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat(
                    (input_ids, next_token_id.unsqueeze(0).unsqueeze(0)), dim=-1
                )
                attention_mask = torch.cat(
                    (attention_mask, torch.tensor([[1]], device=input_ids.device)),
                    dim=-1,
                )

                attention_map = [
                    attention.detach().cpu().half()
                    for attention in output['attentions']
                ]
                attention_map = [
                    torch.nan_to_num(attention, nan=0.0) for attention in attention_map
                ]
                attention_map = get_last_attn(attention_map)
                attention_maps.append(attention_map)

        output_tokens = [
            self.tokenizer.decode(token, skip_special_tokens=True)
            for token in generated_tokens
        ]
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return (
            generated_text,
            output_tokens,
            attention_maps,
            input_tokens,
            data_range,
            generated_probs,
        )
