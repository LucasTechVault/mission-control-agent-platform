# M02 — Tool Execution Boundary

## Architectural Principle

Models propose actions.
Mission Control authorizes and executes actions.

## Model Responsibilities

The model may:

- select an exposed tool
- propose arguments
- decide no tool is necessary
- reason over tool results

The model may not:

- directly access credentials
- execute code
- access networks or databases
- bypass runtime validation
- authorize its own actions

## Runtime Responsibilities

Mission Control runtime owns:

- tool discovery
- tool registry
- argument validation
- execution authorization
- timeout and cancellation
- execution dispatch
- error normalization
- tracing and observability
- tool-result delivery

## Tool Responsibilities

Each tool owns:

- one bounded capability
- a typed input contract
- deterministic argument handling
- execution against its backing system
- normalized output/error behavior

## Core Execution Flow

Model
→ ToolCall
→ Runtime Validation
→ Policy / Authorization
→ Tool Executor
→ ToolResult
→ Model

## Invariants

MC-TOOL-001 through MC-TOOL-010.

```
MC-TOOL-001
The model may request a capability but may never execute it directly.

MC-TOOL-002
Only tools registered by Mission Control may be executed.

MC-TOOL-003
Every tool call must identify one registered tool by exact name.

MC-TOOL-004
Every tool call argument payload must pass deterministic validation
before execution.

MC-TOOL-005
Tool credentials and implementation details are never exposed to the model.

MC-TOOL-006
Every invocation receives a unique call_id for tracing.

MC-TOOL-007
Every invocation produces either a successful ToolResult or a typed ToolError.

MC-TOOL-008
Tool failures are data returned to the runtime; they do not crash the agent loop.

MC-TOOL-009
Tool execution may later be subject to policy, authorization and human approval.

MC-TOOL-010
A schema-valid model request is still not automatically authorized.
```
