import argparse
import json
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent / "probe"))

app = FastAPI()

OLLAMA_BASE_URL = "http://localhost:11434"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

backend_mode = "ollama"
transformer_model = None
transformer_tokenizer = None
probe = None
probe_layer = None


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "qwen2.5:1.5b"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


def generate_with_transformers(
    model, tokenizer, messages: List[Message], temperature: float, max_tokens: Optional[int]
) -> str:
    conversation_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
    conversation_text += "\nassistant:"

    inputs = tokenizer(conversation_text, return_tensors="pt").to(device)

    generation_config = {
        "max_new_tokens": max_tokens if max_tokens else 512,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(conversation_text):].strip()

    return response


async def stream_ollama_response(
    messages: List[Message],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    top_p: Optional[float],
) -> AsyncIterator[str]:
    ollama_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

    request_body = {
        "model": model,
        "messages": ollama_messages,
        "stream": True,
        "options": {
            "temperature": temperature,
        },
    }

    if max_tokens:
        request_body["options"]["num_predict"] = max_tokens
    if top_p:
        request_body["options"]["top_p"] = top_p

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST", f"{OLLAMA_BASE_URL}/api/chat", json=request_body
        ) as response:
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, detail="Ollama API error"
                )

            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        chunk = json.loads(line)

                        if chunk.get("done"):
                            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 0, 'completion_tokens': 0}})}\n\n"
                            yield "data: [DONE]\n\n"
                        else:
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield f"data: {json.dumps({'choices': [{'delta': {'content': content}, 'finish_reason': None}]})}\n\n"
                    except json.JSONDecodeError:
                        continue


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global backend_mode, transformer_model, transformer_tokenizer

    if backend_mode == "ollama":
        if request.stream:
            return StreamingResponse(
                stream_ollama_response(
                    request.messages,
                    request.model,
                    request.temperature,
                    request.max_tokens,
                    request.top_p,
                ),
                media_type="text/event-stream",
            )

        ollama_messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        request_body = {
            "model": request.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }

        if request.max_tokens:
            request_body["options"]["num_predict"] = request.max_tokens
        if request.top_p:
            request_body["options"]["top_p"] = request.top_p

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{OLLAMA_BASE_URL}/api/chat", json=request_body
                )
                response.raise_for_status()
                data = response.json()

                content = data.get("message", {}).get("content", "")

                return {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }
            except httpx.HTTPError as e:
                raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")

    elif backend_mode == "transformers":
        if request.stream:
            raise HTTPException(
                status_code=400, detail="Streaming not supported in transformers mode"
            )

        if transformer_model is None or transformer_tokenizer is None:
            raise HTTPException(
                status_code=500, detail="Transformer model not initialized"
            )

        content = generate_with_transformers(
            transformer_model,
            transformer_tokenizer,
            request.messages,
            request.temperature,
            request.max_tokens,
        )

        response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

        if probe is not None and probe_layer is not None:
            messages_dict = [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ]
            # @TODO: implement injection detection

        return response_data

    else:
        raise HTTPException(status_code=500, detail=f"Unknown backend mode: {backend_mode}")


@app.get("/health")
async def health():
    return {"status": "ok"}


def main():
    global backend_mode, transformer_model, transformer_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ollama", "transformers"],
        default="ollama",
        help="Backend to use for inference (default: ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="Model to use (for transformers backend, default: Qwen/Qwen2.5-1.5B)",
    )

    args = parser.parse_args()
    backend_mode = args.backend

    if backend_mode == "transformers":
        print(f"Loading transformer model: {args.model}")
        transformer_tokenizer = AutoTokenizer.from_pretrained(args.model)
        transformer_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16
        ).to(device)
        transformer_model.eval()
        print(f"Model loaded on {device}")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
