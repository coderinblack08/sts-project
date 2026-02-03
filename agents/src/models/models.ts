import { qLLMProvider } from "./provider.ts";
import { openai } from "@ai-sdk/openai";

export const qLLM = qLLMProvider.languageModel("qwen2.5:1.5b");
export const pLLM = openai("gpt-5-mini");
