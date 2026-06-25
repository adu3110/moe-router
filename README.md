# moe-router

Mixture-of-Experts (MoE) token routing — implemented from scratch in pure NumPy / TypeScript.

Modern LLMs like Mixtral-8x7B and DeepSeek-V2 scale parameters without scaling compute by routing each token to only the top-K relevant expert FFNs. This repo implements the full forward pass so you can trace every number.

## Architecture

```
token embedding (hidden_dim,)
     │
Router  W_r  →  softmax  →  top-K expert selection
     │
Expert FFNs (only K run out of E total)
  expert_i:  h = ReLU(x @ W1[i])   o = h @ W2[i]
     │
Weighted sum of K expert outputs → output embedding
```

Also implements the **Switch Transformer load-balancing loss** to prevent expert collapse.

## Run

```bash
# Python (NumPy only)
python python/moe.py

# TypeScript (zero deps)
npx ts-node typescript/moe.ts
# or: tsc typescript/moe.ts && node typescript/moe.js
# or: deno run typescript/moe.ts
```

## What you'll see

- Per-token routing table with probability bars and selected experts
- Load-balance analysis across the sequence
- Auxiliary loss value (lower = more balanced)

## Key parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `HIDDEN_DIM` | 8 | Token embedding size |
| `FFN_DIM` | 16 | Expert inner dimension |
| `NUM_EXPERTS` | 4 | Total experts |
| `TOP_K` | 2 | Experts activated per token |

## Dependencies

- Python: `numpy` only
- TypeScript: none
