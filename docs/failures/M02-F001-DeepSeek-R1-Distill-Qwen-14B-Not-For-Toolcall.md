Currently serving:

```
deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```

**Problem:**

- DeepSeek R1 series support reasoning and structured JSON output
- Does not support tool calling through its reasoning integration

**Solution:**

- Continue with standard ToolCall code design and implementation
- Do not change architecture to parse raw text instead (messy and unstable)
- change model to 1 that supports Tool Calling.
