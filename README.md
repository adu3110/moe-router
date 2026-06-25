# moe-router

Mixture-of-Experts token routing implemented from scratch. No PyTorch. No Transformers. Pure NumPy (Python) and zero-dependency TypeScript.

Every LLM like Mixtral-8x7B or DeepSeek-V2 uses MoE to run more parameters without paying more compute. This repo shows exactly how it works — small enough to trace every number by hand.

---

## How MoE routing works

A standard FFN runs the same weights for every token. MoE splits those weights into `E` expert networks and sends each token to only the top-`K`:

```
Input token embedding  (hidden_dim,)
         │
         ▼
   Router  W_r              linear projection → (num_experts,)
         │
       softmax              turn logits into probabilities
         │
     top-K select           pick 2 experts out of 4
         │
  ┌──────┴──────┐
  ▼             ▼
Expert 1      Expert 3      each expert: ReLU(x @ W1) @ W2
  │             │
  └──────┬──────┘
         ▼
  weighted sum              output = Σ gate_weight · expert_output
         │
         ▼
  Output embedding  (hidden_dim,)
```

**Why it matters:** 8 experts with top-2 routing gives you 8× the parameters but only 2× the compute per token.

---

## Load-balancing loss

Without a penalty, the router collapses — it learns to always send tokens to the same "winning" expert. The Switch Transformer fix:

```
L_aux = E · Σ_e ( f_e · P_e )

f_e = fraction of tokens routed to expert e   (actual usage)
P_e = mean router probability for expert e    (intended usage)
```

Minimising this during training pushes the router to distribute tokens evenly.

---

## Run

```bash
# Python (numpy only)
pip install numpy
python python/moe.py

# TypeScript (zero dependencies)
npx ts-node typescript/moe.ts
# or: deno run typescript/moe.ts
```

### Sample output

```
Token id=6
  Expert   Prob    Bar         Active
  Expert 0 0.1373  █▁
  Expert 1 0.2339  █▇            ◀ selected
  Expert 2 0.1337  █▁
  Expert 3 0.4951  ████          ◀ selected

  Routing weights (renormalised among top-2):
    Expert 1  ×  0.3209
    Expert 3  ×  0.6791

Expert selection frequency across 6 tokens:
  Expert 0  selected  0x  ▁          0%
  Expert 1  selected  3x  ████▁      50%
  Expert 3  selected  6x  ████████  100%

Auxiliary load-balance loss: 2.33
(lower = more balanced; ideal ≈ 1.0 when perfectly uniform)
```

---

## Parameters

| Name | Default | What it controls |
|------|---------|-----------------|
| `NUM_EXPERTS` | 4 | Total expert networks |
| `TOP_K` | 2 | Experts activated per token |
| `HIDDEN_DIM` | 8 | Token embedding size |
| `FFN_DIM` | 16 | Expert inner dimension |
| `SEQ_LEN` | 6 | Tokens in the demo sequence |

Change these in the first few lines of either file to explore how routing behaviour shifts.

---

## Files

```
moe-router/
├── python/moe.py        pure NumPy — weights, router, expert FFNs, load-balance loss
└── typescript/moe.ts    same logic, zero npm dependencies
```

---

## Related concepts

- **Mixtral-8x7B** — 8 experts, top-2 routing, 46.7B total params, 12.9B active per token
- **DeepSeek-V2** — 160 experts, top-6 routing
- **Switch Transformer** (Fedus et al., 2022) — introduced the load-balancing loss implemented here
