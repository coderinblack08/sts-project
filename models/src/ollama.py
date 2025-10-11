import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

OLLAMA_BASE_URL = "http://localhost:11434"


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


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("Starting server on http://localhost:8000")
    print("Make sure Ollama is running with: ollama run qwen2.5:1.5b")
    uvicorn.run(app, host="0.0.0.0", port=8000)
