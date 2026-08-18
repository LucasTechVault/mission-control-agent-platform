MC-AGENT-001
Mission Control owns the execution loop.

MC-AGENT-002
The model chooses between proposing actions and returning a final answer.

MC-AGENT-003
A ToolCall remains an untrusted proposal and must cross the M02 executor boundary.

MC-AGENT-004
ToolResult is an observation, not an automatic final answer.

MC-AGENT-005
Tool failures are returned to model as observations when safe to do so.

MC-AGENT-006
The model never invokes itself. Mission Control decides whether another model turn occurs.

MC-AGENT-007
The runtime must eventually impose deterministic loop termination rules independent of model preference.

MC-AGENT-008
The runtime (application holds the absolute truth) in memory (the state). LLM only sees a stringified, curated summary of the truth (context).

MC-AGENT-009
One model turn will be zero or one tool.

MC-AGENT-010
A model turn with no tool calls and valid final text terminates the initial manual loop.

## Typical Production Architecture for Agent Runtime

```
                        USER OBJECTIVE
                              │
                              ▼
                        Runtime State
                              │
                              ▼
                       Context Builder
                              │
                              ▼
                           MODEL
                              │
                       structured output
                     /                  \
                    /                    \
              ToolCall(s)              Final
                  │                       │
                  ▼                       ▼
              Validation                DONE
                  │
              Authorization
                  │
               Executor
                  │
              ToolResult
                  │
              Observation
                  │
              State update
                  │
            Budget / policy
                  │
             another turn
```
