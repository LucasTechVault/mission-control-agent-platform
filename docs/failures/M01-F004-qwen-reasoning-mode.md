```
When serving Qwen3 on vLLM, forgot to add flags:
    --enable-reasoning
    --reasoning-parser deepseek_r1

This resulted reasoning tokens to be added to output token.

WITHOUT PARSER:
content = [reasoning + final answer]

WITH PARSER:

reasoning = [reasoning]
content = [final answer]


```

{
"role": "system",
"message": "Return answer as a JSON object with a severity key."
}
