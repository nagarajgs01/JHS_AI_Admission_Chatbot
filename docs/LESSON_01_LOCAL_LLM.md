# Lesson 1: Connecting FastAPI to a self-hosted LLM

## Learning goal

Understand why the application talks to a model through an adapter instead of
placing Ollama-specific code inside the chat endpoint.

## The boundary

The chat service depends on the `LanguageModel` protocol. Three adapters implement it:

- `EvidenceOnlyModel`: deterministic automated tests;
- `OllamaModel`: local Mac development;
- `VLLMModel`: production GPU serving.

The rest of the application does not care which runtime is selected. Environment
variables choose the implementation.

## Run Qwen locally

Install Ollama, then run:

```bash
ollama pull qwen3:4b
ollama run qwen3:4b
```

In another terminal, verify the local server:

```bash
curl http://localhost:11434/api/show \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:4b"}'
```

## Configure the API

Copy `.env.example` to `.env`. The important values are:

```dotenv
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3:4b
```

`LLM_PROVIDER=mock` runs the backend without Ollama and is useful for tests.

## What is sent to Qwen

The adapter sends a system safety policy followed by the approved retrieved passages
and the parent's question. Lead names, phone numbers, and email addresses are handled
by a separate endpoint and are never included in the model request.

## Why temperature is zero

This chatbot values consistent, grounded answers over creativity. Temperature zero
reduces variation, but it does not guarantee factuality. Retrieval thresholds,
source restrictions, validation, and refusal behavior remain necessary.

## Exercise

Run the API once with `LLM_PROVIDER=mock`, and once with `LLM_PROVIDER=ollama`.
Ask:

1. What grades are offered?
2. Does the school provide swimming facilities?
3. What is the Grade 5 fee?

The first two should use brochure evidence. The third must be escalated before the
model is called because the approved knowledge base contains no fee information.
