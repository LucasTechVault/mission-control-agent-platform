# M01 - Local LLM Runtime Architecture

## Purpose

Mission Control separates deterministic app logic from probabilistic model inference.

The model is treated as an **external computation service** rather than being embedded directly in app logic.

## Logical Architecture

```
User / CLI
    ↓
Mission Control Application
    ↓
ModelGateway
    ↓
   HTTP
    ↓
vLLM Inference Server
    ↓
Open-Weight Model
    ↓
   GPU
```

## Responsibilities

### Mission Control (Backend App)

Owns:

- user input validation
- message construction
- app config
- request lifecycle
- response validation
- logging
- app decisions

### ModelGateway (Contract, not running app)

- It is Abstract Base Class (Protocol in modern Python)
- Defines the rules for what a model MUST do
- Doesn't care how the model does it.

### vLLM

Owns the model inference runtime:

- model loading
- inference API
- tokenization / chat rendering
- request scheduling
- KV-cache management
- decoding
- streaming
