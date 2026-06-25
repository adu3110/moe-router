"""
Mixture-of-Experts (MoE) Router — implemented from scratch.

What this demonstrates
----------------------
Modern LLMs like Mixtral-8x7B and DeepSeek-V2 use MoE to scale parameters
without scaling compute. Instead of running every weight for every token,
a learned "router" sends each token to only the top-K relevant expert FFNs.
This file implements the full forward pass in ~200 lines of pure NumPy so you
can trace every number.

Architecture
------------
    token embedding (hidden_dim,)
         │
    Router (linear + softmax)
         │
    top-K expert selection + load-balancing
         │
    Expert FFNs run in parallel (linear → ReLU → linear)
         │
    Weighted sum of expert outputs
         │
    output embedding (hidden_dim,)

Run
---
    python moe.py

Dependencies: numpy only.
"""

from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters — small enough to inspect every number by hand
# ──────────────────────────────────────────────────────────────────────────────
HIDDEN_DIM   = 8    # token embedding size
FFN_DIM      = 16   # inner dimension of each expert FFN
NUM_EXPERTS  = 4    # total experts in the mixture
TOP_K        = 2    # how many experts each token uses
VOCAB_SIZE   = 16   # tiny vocabulary for the demo
SEQ_LEN      = 6    # number of tokens in the demo sequence

RNG = np.random.default_rng(42)


# ──────────────────────────────────────────────────────────────────────────────
# Weight initialisation  (Xavier uniform, same as PyTorch defaults)
# ──────────────────────────────────────────────────────────────────────────────
def xavier(rows: int, cols: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (rows + cols))
    return RNG.uniform(-limit, limit, (rows, cols))


class Weights:
    """All learnable parameters — one flat namespace so nothing is hidden."""

    def __init__(self) -> None:
        # Token embedding table  (VOCAB_SIZE × HIDDEN_DIM)
        self.embed: np.ndarray = xavier(VOCAB_SIZE, HIDDEN_DIM)

        # Router: projects token → logit per expert  (HIDDEN_DIM × NUM_EXPERTS)
        self.router: np.ndarray = xavier(HIDDEN_DIM, NUM_EXPERTS)

        # Expert FFNs: each is a two-layer network
        # W1: (NUM_EXPERTS, HIDDEN_DIM, FFN_DIM)
        # W2: (NUM_EXPERTS, FFN_DIM, HIDDEN_DIM)
        self.expert_w1: np.ndarray = np.stack(
            [xavier(HIDDEN_DIM, FFN_DIM) for _ in range(NUM_EXPERTS)]
        )
        self.expert_w2: np.ndarray = np.stack(
            [xavier(FFN_DIM, HIDDEN_DIM) for _ in range(NUM_EXPERTS)]
        )


# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────
def route(token: np.ndarray, w: Weights) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given one token embedding, return:
      logits      — raw router scores (NUM_EXPERTS,)
      probs       — softmax over all experts (NUM_EXPERTS,)
      top_k_idx   — indices of the TOP_K chosen experts (TOP_K,)
    """
    logits: np.ndarray = token @ w.router               # (NUM_EXPERTS,)

    # Softmax: shift by max for numerical stability
    shifted  = logits - logits.max()
    exp_vals = np.exp(shifted)
    probs: np.ndarray = exp_vals / exp_vals.sum()       # (NUM_EXPERTS,)

    top_k_idx: np.ndarray = np.argsort(probs)[-TOP_K:]  # (TOP_K,) highest prob

    return logits, probs, top_k_idx


# ──────────────────────────────────────────────────────────────────────────────
# Expert FFN
# ──────────────────────────────────────────────────────────────────────────────
def expert_ffn(token: np.ndarray, expert_idx: int, w: Weights) -> np.ndarray:
    """
    Run one expert's two-layer FFN on a single token.

    forward pass:
        h = ReLU(token @ W1[e])   # up-projection
        o = h @ W2[e]             # down-projection
    """
    h: np.ndarray = np.maximum(0.0, token @ w.expert_w1[expert_idx])  # ReLU
    return h @ w.expert_w2[expert_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Full MoE forward pass for one token
# ──────────────────────────────────────────────────────────────────────────────
def moe_forward(token: np.ndarray, w: Weights) -> tuple[np.ndarray, dict]:
    """
    MoE forward pass for a single token embedding.

    Returns
    -------
    output : np.ndarray  (HIDDEN_DIM,)
        Weighted sum of the top-K expert outputs.
    trace : dict
        Every intermediate value so you can inspect the routing decision.
    """
    logits, probs, top_k_idx = route(token, w)

    # Renormalise weights among only the chosen experts
    chosen_probs = probs[top_k_idx]
    chosen_probs = chosen_probs / chosen_probs.sum()  # re-normalise to sum=1

    output = np.zeros(HIDDEN_DIM)
    expert_outputs: dict[int, np.ndarray] = {}

    for weight, eidx in zip(chosen_probs, top_k_idx):
        expert_out = expert_ffn(token, int(eidx), w)
        expert_outputs[int(eidx)] = expert_out
        output += weight * expert_out             # weighted mixture

    trace = {
        "router_logits": logits,
        "router_probs":  probs,
        "top_k_experts": top_k_idx.tolist(),
        "chosen_weights": chosen_probs.tolist(),
        "expert_outputs": expert_outputs,
    }
    return output, trace


# ──────────────────────────────────────────────────────────────────────────────
# Load-balancing loss
# ──────────────────────────────────────────────────────────────────────────────
def load_balance_loss(all_probs: np.ndarray) -> float:
    """
    Switch Transformer auxiliary loss that penalises expert collapse.

    Without this, the router tends to send everything to one "winning" expert
    because gradient descent reinforces popular experts. The loss nudges the
    router to distribute tokens more evenly.

    formula:  L = NUM_EXPERTS · Σ_e ( f_e · P_e )
      f_e = fraction of tokens routed to expert e
      P_e = mean router probability for expert e across all tokens
    """
    T = all_probs.shape[0]                       # num tokens
    top_k_indices = np.argsort(all_probs, axis=1)[:, -TOP_K:]

    # f_e: how many tokens chose each expert (normalised)
    counts = np.zeros(NUM_EXPERTS)
    for row in top_k_indices:
        counts[row] += 1
    f_e = counts / T                             # (NUM_EXPERTS,)

    # P_e: mean router probability across all tokens
    P_e = all_probs.mean(axis=0)                 # (NUM_EXPERTS,)

    return float(NUM_EXPERTS * np.dot(f_e, P_e))


# ──────────────────────────────────────────────────────────────────────────────
# Visualiser
# ──────────────────────────────────────────────────────────────────────────────
_BARS = "▁▂▃▄▅▆▇█"

def _bar(prob: float, width: int = 8) -> str:
    filled = int(prob * width)
    partial = int((prob * width - filled) * len(_BARS))
    bar = "█" * filled
    if filled < width:
        bar += _BARS[partial]
    return bar.ljust(width)


def print_routing_table(token_id: int, trace: dict) -> None:
    probs = trace["router_probs"]
    chosen = set(trace["top_k_experts"])
    print(f"\n  Token id={token_id}")
    print(f"  {'Expert':<8} {'Prob':>6}  {'Bar':<10}  {'Active'}")
    print(f"  {'──────':<8} {'────':>6}  {'───':<10}  {'──────'}")
    for e, p in enumerate(probs):
        active = "  ◀ selected" if e in chosen else ""
        print(f"  Expert {e}   {p:.4f}  {_bar(p):<10}{active}")
    print(f"\n  Routing weights (renormalised among top-{TOP_K}):")
    for eidx, w in zip(trace["top_k_experts"], trace["chosen_weights"]):
        print(f"    Expert {eidx}  ×  {w:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main demo
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  Mixture-of-Experts Router — from scratch (pure NumPy)")
    print(f"  {NUM_EXPERTS} experts · top-{TOP_K} routing · hidden={HIDDEN_DIM}")
    print("=" * 60)

    w = Weights()

    # Simulate a small sequence of token ids
    token_ids = RNG.integers(0, VOCAB_SIZE, SEQ_LEN).tolist()
    print(f"\nInput token ids: {token_ids}")

    all_outputs: list[np.ndarray] = []
    all_probs:   list[np.ndarray] = []

    for tid in token_ids:
        embedding = w.embed[tid]                       # lookup
        output, trace = moe_forward(embedding, w)
        all_outputs.append(output)
        all_probs.append(trace["router_probs"])
        print_routing_table(tid, trace)

    all_probs_arr = np.array(all_probs)   # (SEQ_LEN, NUM_EXPERTS)

    print("\n" + "=" * 60)
    print("  Load-balancing analysis")
    print("=" * 60)
    print(f"\n  Expert selection frequency across {SEQ_LEN} tokens:")
    top_k_indices = np.argsort(all_probs_arr, axis=1)[:, -TOP_K:]
    counts = np.zeros(NUM_EXPERTS, dtype=int)
    for row in top_k_indices:
        counts[row] += 1
    for e, c in enumerate(counts):
        pct = c / SEQ_LEN * 100
        print(f"    Expert {e}  selected {c:2d}x  {_bar(pct/100):<10} {pct:.0f}%")

    lb_loss = load_balance_loss(all_probs_arr)
    print(f"\n  Auxiliary load-balance loss: {lb_loss:.4f}")
    print("  (lower = more balanced; ideal ≈ 1.0 when perfectly uniform)")

    print("\n" + "=" * 60)
    print("  Output shape:", np.array(all_outputs).shape)
    print("  First output vector (truncated):", all_outputs[0].round(4))
    print("=" * 60)


if __name__ == "__main__":
    main()
