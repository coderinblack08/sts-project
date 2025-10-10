# Usage

This project demonstrates a custom AI SDK provider that connects to a Python FastAPI backend, which proxies requests to Ollama for local LLM inference.

## Architecture

```
TypeScript Client (AI SDK) → Python FastAPI Server → Ollama
```

## Components

### 1. Python FastAPI Server (`models/main.py`)

A FastAPI server that:

- Exposes OpenAI-compatible `/v1/chat/completions` endpoint
- Proxies requests to local Ollama instance
- Supports both streaming and non-streaming responses
- Uses `qwen2.5:1.5b` model (small, fast, efficient)

**Run the server:**

```bash
cd models
uv run python main.py
```

Server runs on `http://localhost:8000`

### 2. TypeScript Custom Provider (`agents/src/customProvider.ts`)

A custom AI SDK provider that:

- Implements AI SDK v2 specification
- Supports both `doGenerate()` and `doStream()` methods
- Uses `customProvider` with `fallbackProvider` for dynamic model support
- Connects to the Python API endpoint

### 3. Main Export (`agents/src/main.ts`)

Exports pre-configured models for convenient usage:

- `quantizedModel` - Uses `qwen2.5:1.5b` (small, fast)
- `largerModel` - Uses `llama3.1:8b` (larger, more capable)

The model ID is now **actually passed through** to Ollama, so you can use any available Ollama model!

**Usage:**

```typescript
import { generateText } from "ai";
import { quantizedModel } from "./main.js";

const result = await generateText({
  model: quantizedModel,
  prompt: "What is 2+2?",
  temperature: 0.7,
});

console.log(result.text);
```

Or use the provider directly with any Ollama model:

```typescript
import { ollamaProvider } from "./customProvider.js";

const customModel = ollamaProvider.languageModel("llama3.1:8b");
const smallModel = ollamaProvider.languageModel("qwen2.5:1.5b");
const anyModel = ollamaProvider.languageModel("your-model-name");
```
