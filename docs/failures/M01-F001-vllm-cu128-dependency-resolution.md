## M01 - Dependency Resolution Failure

```
We wanted:
vLLM 0.26
    +
CUDA 12.8

Resolver discovered:
vLLM 0.26
    │
    └── torchcodec >= 0.14

CUDA 12.8 PyTorch ecosystem
    │
    └── torchcodec <= 0.11.1
             ↓
       UNSATISFIABLE
```

**Actions not to do:**
❌ --no-deps
❌ force-install random packages
❌ edit torchcodec requirement
❌ mix CUDA 13 libraries into CUDA 12.8
❌ upgrade host driver blindly

**Resolution Steps:**
Hardware / driver constraint
↓
CUDA 12.8
↓
choose compatible PyTorch
↓
choose Qwen-supported vLLM release
↓
build runtime against known stack

**Key Engineering Principle:**

> Dependency resolvers are not obstacles to bypass.
> They prove that proposed software stack is internally inconsistent.
