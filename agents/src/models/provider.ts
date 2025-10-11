import { customProvider } from "ai";

interface CustomProviderSettings {
  baseURL?: string;
  apiKey?: string;
}

interface OllamaMessage {
  role: string;
  content: string;
}

function convertToMessages(prompt: any[]): OllamaMessage[] {
  return prompt.map((message: any) => {
    if (message.role === "system") {
      return { role: "system", content: message.content };
    }
    if (message.role === "user") {
      const content = message.content
        .map((part: any) => {
          if (part.type === "text") return part.text;
          return "";
        })
        .join("");
      return { role: "user", content };
    }
    if (message.role === "assistant") {
      const content = message.content
        .map((part: any) => {
          if (part.type === "text") return part.text;
          return "";
        })
        .join("");
      return { role: "assistant", content };
    }
    return { role: message.role, content: "" };
  });
}

export function createOllamaProvider(settings: CustomProviderSettings = {}) {
  const baseURL = settings.baseURL || "http://localhost:8000";
  const apiKey = settings.apiKey;

  const provider = customProvider({
    languageModels: {},
    fallbackProvider: {
      languageModel: (modelId: string) => ({
        specificationVersion: "v2" as const,
        provider: "ollama",
        modelId,
        defaultObjectGenerationMode: "tool" as const,

        async doGenerate(options: any) {
          const messages = convertToMessages(options.prompt);

          const requestBody = {
            model: modelId,
            messages,
            temperature: options.temperature,
            max_tokens: options.maxTokens,
            top_p: options.topP,
            stream: false,
          };

          const headers: Record<string, string> = {
            "Content-Type": "application/json",
          };

          if (apiKey) {
            headers["Authorization"] = `Bearer ${apiKey}`;
          }

          const response = await fetch(`${baseURL}/v1/chat/completions`, {
            method: "POST",
            headers,
            body: JSON.stringify(requestBody),
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const data = await response.json();
          const text = data.choices?.[0]?.message?.content || "";

          return {
            content: [{ type: "text", text }],
            finishReason: data.choices?.[0]?.finish_reason || "stop",
            usage: {
              promptTokens: data.usage?.prompt_tokens || 0,
              completionTokens: data.usage?.completion_tokens || 0,
              totalTokens:
                (data.usage?.prompt_tokens || 0) +
                (data.usage?.completion_tokens || 0),
            },
            rawCall: { rawPrompt: requestBody, rawSettings: {} },
            warnings: [],
          };
        },

        async doStream(options: any) {
          const messages = convertToMessages(options.prompt);

          const requestBody = {
            model: modelId,
            messages,
            temperature: options.temperature,
            max_tokens: options.maxTokens,
            top_p: options.topP,
            stream: true,
          };

          const headers: Record<string, string> = {
            "Content-Type": "application/json",
          };

          if (apiKey) {
            headers["Authorization"] = `Bearer ${apiKey}`;
          }

          const response = await fetch(`${baseURL}/v1/chat/completions`, {
            method: "POST",
            headers,
            body: JSON.stringify(requestBody),
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error("Response body is not readable");
          }

          const decoder = new TextDecoder();
          let buffer = "";

          const stream = new ReadableStream({
            async start(controller) {
              let finished = false;
              try {
                while (true) {
                  const { done, value } = await reader.read();

                  if (done) {
                    break;
                  }

                  buffer += decoder.decode(value, { stream: true });
                  const lines = buffer.split("\n");
                  buffer = lines.pop() || "";

                  for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || !trimmed.startsWith("data: ")) continue;

                    const data = trimmed.slice(6);
                    if (data === "[DONE]") {
                      finished = true;
                      break;
                    }

                    try {
                      const parsed = JSON.parse(data);
                      const delta = parsed.choices?.[0]?.delta;
                      const finishReason = parsed.choices?.[0]?.finish_reason;

                      if (delta?.content) {
                        controller.enqueue({
                          type: "text-delta",
                          textDelta: delta.content,
                        });
                      }

                      if (finishReason) {
                        const promptTokens = parsed.usage?.prompt_tokens || 0;
                        const completionTokens =
                          parsed.usage?.completion_tokens || 0;
                        controller.enqueue({
                          type: "finish",
                          finishReason: finishReason || "stop",
                          usage: {
                            promptTokens,
                            completionTokens,
                            totalTokens: promptTokens + completionTokens,
                          },
                        });
                        finished = true;
                        break;
                      }
                    } catch (e) {
                      console.error("Error parsing SSE data:", e);
                    }
                  }

                  if (finished) break;
                }

                controller.close();
              } catch (error) {
                controller.error(error);
              }
            },
          });

          return {
            stream,
            rawCall: { rawPrompt: requestBody, rawSettings: {} },
            warnings: [],
          };
        },
      }),
    } as any,
  });

  return provider;
}

export const ollamaProvider = createOllamaProvider();

/*
Usage Example:

import { streamText } from 'ai';
import { ollamaProvider } from './main.js';

const result = await streamText({
  model: ollamaProvider.languageModel('qwen2.5:1.5b'),
  prompt: 'Tell me a story',
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}

Custom Provider with different baseURL:

const customOllamaProvider = createOllamaProvider({
  baseURL: 'http://localhost:5000',
  apiKey: 'optional-api-key',
});

const model = customOllamaProvider.languageModel('qwen2.5:1.5b');
*/
