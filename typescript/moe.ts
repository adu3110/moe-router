/**
 * Mixture-of-Experts (MoE) Router — implemented from scratch.
 *
 * What this demonstrates
 * ----------------------
 * Modern LLMs like Mixtral-8x7B and DeepSeek-V2 use MoE to scale parameters
 * without scaling compute. A learned "router" sends each token to only the
 * top-K relevant expert FFNs. This file implements the full forward pass with
 * zero dependencies so you can trace every number.
 *
 * Architecture
 * ------------
 *   token embedding  →  Router (linear + softmax)  →  top-K selection
 *   →  Expert FFNs (linear → ReLU → linear)  →  weighted sum  →  output
 *
 * Run
 * ---
 *   npx ts-node moe.ts          (with ts-node)
 *   tsc moe.ts && node moe.js   (compiled)
 *   deno run moe.ts             (Deno)
 *
 * Dependencies: none.
 */

// ──────────────────────────────────────────────────────────────────────────────
// Hyperparameters
// ──────────────────────────────────────────────────────────────────────────────
const HIDDEN_DIM  = 8;
const FFN_DIM     = 16;
const NUM_EXPERTS = 4;
const TOP_K       = 2;
const VOCAB_SIZE  = 16;
const SEQ_LEN     = 6;

// ──────────────────────────────────────────────────────────────────────────────
// Minimal linear-algebra helpers (pure TypeScript, no deps)
// ──────────────────────────────────────────────────────────────────────────────
type Vec = number[];
type Mat = number[][];

/** Dot product of two vectors */
function dot(a: Vec, b: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Matrix-vector multiply: (rows × cols) mat · (cols,) vec → (rows,) vec */
function matvec(mat: Mat, vec: Vec): Vec {
  return mat.map(row => dot(row, vec));
}

/** Transpose a matrix */
function transpose(mat: Mat): Mat {
  const rows = mat.length, cols = mat[0].length;
  return Array.from({ length: cols }, (_, j) =>
    Array.from({ length: rows }, (_, i) => mat[i][j])
  );
}

/** Softmax — numerically stable */
function softmax(logits: Vec): Vec {
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

/** Element-wise ReLU */
function relu(v: Vec): Vec {
  return v.map(x => Math.max(0, x));
}

/** Indices of the K largest values */
function topKIndices(arr: Vec, k: number): number[] {
  return arr
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .slice(0, k)
    .map(e => e.i);
}

// ──────────────────────────────────────────────────────────────────────────────
// Seeded pseudo-random (LCG) — reproducible without any library
// ──────────────────────────────────────────────────────────────────────────────
class LCG {
  private state: number;
  constructor(seed = 42) { this.state = seed; }

  next(): number {
    this.state = (1664525 * this.state + 1013904223) & 0xffffffff;
    return (this.state >>> 0) / 0x100000000;
  }

  /** Xavier uniform initialiser */
  xavier(rows: number, cols: number): Mat {
    const limit = Math.sqrt(6 / (rows + cols));
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (this.next() * 2 - 1) * limit)
    );
  }

  integers(n: number, max: number): number[] {
    return Array.from({ length: n }, () => Math.floor(this.next() * max));
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Model weights
// ──────────────────────────────────────────────────────────────────────────────
interface Weights {
  embed:     Mat;          // VOCAB_SIZE × HIDDEN_DIM
  router:    Mat;          // HIDDEN_DIM × NUM_EXPERTS  (transposed for matvec)
  expertW1:  Mat[];        // NUM_EXPERTS × (HIDDEN_DIM × FFN_DIM)
  expertW2:  Mat[];        // NUM_EXPERTS × (FFN_DIM × HIDDEN_DIM)
}

function initWeights(rng: LCG): Weights {
  return {
    embed:    rng.xavier(VOCAB_SIZE, HIDDEN_DIM),
    router:   transpose(rng.xavier(HIDDEN_DIM, NUM_EXPERTS)),
    expertW1: Array.from({ length: NUM_EXPERTS }, () =>
                transpose(rng.xavier(HIDDEN_DIM, FFN_DIM))),
    expertW2: Array.from({ length: NUM_EXPERTS }, () =>
                transpose(rng.xavier(FFN_DIM, HIDDEN_DIM))),
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Router
// ──────────────────────────────────────────────────────────────────────────────
interface RouteResult {
  logits:    Vec;
  probs:     Vec;
  topKIdx:   number[];
}

function route(token: Vec, w: Weights): RouteResult {
  // W_router is stored transposed so each row is one expert's weight vector
  const logits  = w.router.map(row => dot(row, token));  // (NUM_EXPERTS,)
  const probs   = softmax(logits);
  const topKIdx = topKIndices(probs, TOP_K);
  return { logits, probs, topKIdx };
}

// ──────────────────────────────────────────────────────────────────────────────
// Expert FFN
// ──────────────────────────────────────────────────────────────────────────────
function expertFFN(token: Vec, eidx: number, w: Weights): Vec {
  // up-projection: token (HIDDEN_DIM,) → h (FFN_DIM,)
  const h = relu(w.expertW1[eidx].map(row => dot(row, token)));
  // down-projection: h (FFN_DIM,) → out (HIDDEN_DIM,)
  return w.expertW2[eidx].map(row => dot(row, h));
}

// ──────────────────────────────────────────────────────────────────────────────
// MoE forward pass
// ──────────────────────────────────────────────────────────────────────────────
interface MoETrace {
  routerLogits:   Vec;
  routerProbs:    Vec;
  topKExperts:    number[];
  chosenWeights:  number[];
  expertOutputs:  Record<number, Vec>;
}

function moeForward(token: Vec, w: Weights): { output: Vec; trace: MoETrace } {
  const { logits, probs, topKIdx } = route(token, w);

  // Renormalise weights among only the chosen experts
  const chosenRaw  = topKIdx.map(i => probs[i]);
  const chosenSum  = chosenRaw.reduce((a, b) => a + b, 0);
  const chosenW    = chosenRaw.map(p => p / chosenSum);

  const output: Vec = new Array(HIDDEN_DIM).fill(0);
  const expertOutputs: Record<number, Vec> = {};

  for (let k = 0; k < TOP_K; k++) {
    const eidx = topKIdx[k];
    const expOut = expertFFN(token, eidx, w);
    expertOutputs[eidx] = expOut;
    for (let d = 0; d < HIDDEN_DIM; d++) {
      output[d] += chosenW[k] * expOut[d];
    }
  }

  return {
    output,
    trace: {
      routerLogits:  logits,
      routerProbs:   probs,
      topKExperts:   topKIdx,
      chosenWeights: chosenW,
      expertOutputs,
    },
  };
}

// ──────────────────────────────────────────────────────────────────────────────
// Load-balancing auxiliary loss
// ──────────────────────────────────────────────────────────────────────────────
function loadBalanceLoss(allProbs: Mat): number {
  // allProbs: (T, NUM_EXPERTS)
  const T = allProbs.length;
  const counts = new Array(NUM_EXPERTS).fill(0);
  for (const probs of allProbs) {
    const chosen = topKIndices(probs, TOP_K);
    chosen.forEach(i => counts[i]++);
  }
  const f_e = counts.map(c => c / T);
  const P_e = Array.from({ length: NUM_EXPERTS }, (_, e) =>
    allProbs.reduce((s, row) => s + row[e], 0) / T
  );
  return NUM_EXPERTS * f_e.reduce((s, f, e) => s + f * P_e[e], 0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Visualiser
// ──────────────────────────────────────────────────────────────────────────────
const BARS = "▁▂▃▄▅▆▇█";

function barChart(prob: number, width = 8): string {
  const filled  = Math.floor(prob * width);
  const partial = Math.floor((prob * width - filled) * BARS.length);
  let bar = "█".repeat(filled);
  if (filled < width) bar += BARS[partial] ?? " ";
  return bar.padEnd(width);
}

function printRoutingTable(tokenId: number, trace: MoETrace): void {
  const chosen = new Set(trace.topKExperts);
  console.log(`\n  Token id=${tokenId}`);
  console.log(`  ${"Expert".padEnd(8)} ${"Prob".padStart(6)}  ${"Bar".padEnd(10)}  Active`);
  console.log(`  ${"──────".padEnd(8)} ${"────".padStart(6)}  ${"───".padEnd(10)}  ──────`);
  trace.routerProbs.forEach((p, e) => {
    const active = chosen.has(e) ? "  ◀ selected" : "";
    console.log(`  Expert ${e}   ${p.toFixed(4)}  ${barChart(p).padEnd(10)}${active}`);
  });
  console.log(`\n  Routing weights (renormalised among top-${TOP_K}):`);
  trace.topKExperts.forEach((eidx, k) => {
    console.log(`    Expert ${eidx}  ×  ${trace.chosenWeights[k].toFixed(4)}`);
  });
}

// ──────────────────────────────────────────────────────────────────────────────
// Main demo
// ──────────────────────────────────────────────────────────────────────────────
function main(): void {
  const HR = "=".repeat(60);
  console.log(HR);
  console.log("  Mixture-of-Experts Router — from scratch (pure TypeScript)");
  console.log(`  ${NUM_EXPERTS} experts · top-${TOP_K} routing · hidden=${HIDDEN_DIM}`);
  console.log(HR);

  const rng      = new LCG(42);
  const w        = initWeights(rng);
  const tokenIds = rng.integers(SEQ_LEN, VOCAB_SIZE);

  console.log(`\nInput token ids: [${tokenIds.join(", ")}]`);

  const allOutputs: Vec[] = [];
  const allProbs:   Mat   = [];

  for (const tid of tokenIds) {
    const embedding      = w.embed[tid];
    const { output, trace } = moeForward(embedding, w);
    allOutputs.push(output);
    allProbs.push(trace.routerProbs);
    printRoutingTable(tid, trace);
  }

  console.log("\n" + HR);
  console.log("  Load-balancing analysis");
  console.log(HR);
  console.log(`\n  Expert selection frequency across ${SEQ_LEN} tokens:`);
  const counts = new Array(NUM_EXPERTS).fill(0);
  for (const probs of allProbs) {
    topKIndices(probs, TOP_K).forEach(i => counts[i]++);
  }
  counts.forEach((c, e) => {
    const pct = (c / SEQ_LEN) * 100;
    console.log(`    Expert ${e}  selected ${c.toString().padStart(2)}x  ${barChart(pct / 100).padEnd(10)} ${pct.toFixed(0)}%`);
  });

  const lb = loadBalanceLoss(allProbs);
  console.log(`\n  Auxiliary load-balance loss: ${lb.toFixed(4)}`);
  console.log("  (lower = more balanced; ideal ≈ 1.0 when perfectly uniform)");

  console.log("\n" + HR);
  console.log(`  Output shape: [${SEQ_LEN}, ${HIDDEN_DIM}]`);
  console.log("  First output vector:", allOutputs[0].map(x => x.toFixed(4)).join("  "));
  console.log(HR);
}

main();
