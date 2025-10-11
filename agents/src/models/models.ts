import { ollamaProvider } from "./provider.ts";
import { openai } from "@ai-sdk/openai";

export const quarantinedModel = ollamaProvider.languageModel("qwen2.5:1.5b");

export const privilegedModel = openai("gpt-5-mini");
