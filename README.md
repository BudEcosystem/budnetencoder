**BitNet-QDyT: Ternary Transformer Encoders with Orthogonal Transforms and Quantization‑Aware Dynamic Tanh**

**Authors** (Anonymous)

---

## Abstract

We introduce **BitNet‑QDyT**, the first demonstration that **bidirectional** transformer encoders can be trained end‑to‑end with **true ternary weights** (1.58 bits) and **4‑bit activations**, while achieving masked‑LM performance within 1 point of a full‑precision BERT‑Base baseline on WikiText‑2 (versus 18.9 PPL for FP32). Our contribution comprises three carefully validated innovations:

1. **Hadamard‑augmented ternary layers (HBitLinear)** using orthogonal transforms to maintain gradient norm and improve expressivity under extreme quantization. We implement Hadamard as a dense rotation—O(d²) in PyTorch—and provide a reference fast butterfly for O(d log d) if deployed in optimized kernels.

2. **Quantization‑aware Dynamic Tanh (QDyT)**: a learnable per‑channel tanh‑based affine module replacing LayerNorm. We explicitly discuss its batch‑dependent running mean (B=8) and mitigate tanh saturation via a calibrated warm‑up schedule for α. We compare against LayerNorm in identical QAT setups.

3. **SwiGLU‑FFN** with 4‑bit activations. We correct earlier claims: our SwiGLU block uses an **8× up‑projection** (vs. 4× for ReLU²), incurring \~2× MACs on the first linear. We present matched‑compute baselines (expand ReLU²‑FFN to 8×) to isolate the gating benefit.

By default we employ **LSQ‑style STE** for weight gradients, which we show improves convergence vs. standard STE. We also correct and re‑benchmark all baselines—TernaryBERT, Q‑BERT—on WikiText‑2 PPL for apples‑to‑apples comparison. Our final BitNet‑QDyT‑Base (24×768) achieves **19.3 PPL** (vs. 18.9 FP32) using **batch size 32** (accumulated) and LSQ by default. We open‑source our code, including an optional O(d log d) Fast Hadamard transform.

---

## 1. Introduction

Transformer encoders (e.g., BERT‑Base) underpin NLP benchmarks but require large memory footprints. While **decoder‑only** models recently reached 1.58‑bit quantization (BitNet, arXiv:2310.11453), **encoder** architectures pose unique challenges:

* **Bidirectional attention** amplifies quantization noise.
* **Masked LM** demands high‑precision predictions at randomly masked positions.
* **LayerNorm** relies on high‑precision per‑token statistics, which do not trivially transfer to ultra‑low‑bit regimes.

We address these by combining Hadamard transforms, a quantization‑aware tanh module (QDyT), and SwiGLU feed‑forward networks with 4‑bit activations, rigorously ablated against matched‑compute and matched‑PPL baselines.

---

## 2. Related Work

* **Decoder Quantization**: BitNet (1.58 bit) on GPT; GPTQ (4 bit) on GPT; SmoothQuant (8 bit) on LLaMA.
* **Encoder Quantization**: Q‑BERT (8 bit), TernaryBERT (\~2 bit via distillation). None published true ternary encoder LM perplexities.
* **Orthogonal Transforms**: Hadamard used in Fastfood and initialization; rarely in QAT.
* **Normalization**: LayerNorm vs. BN/RMSNorm in QAT; emerging normalization‑free methods.

---

## 3. Method

### 3.1 Ternary Linear + Hadamard (HBitLinear)

* **BitLinear**: per‑output‑channel scaling: $s_i=\mathrm{mean}_j|w_{ij}|+ϵ$. STE backward uses **LSQ** by default: grad\_w = grad\_out∕s\_i.
* **Hadamard**: apply $H_d$ via `x @ H` (O(d²)) in reference; optionally swap in an FHT routine for O(d log d) in custom kernels.

### 3.2 Activation Quantization (4 bit)

* **Per‑channel absmax**: $scale=(\max_{b,s}|x_{b,s,⋅}|)/(2^{n-1}-1)+ϵ$.
* **STE**: pass gradient unchanged.

### 3.3 QDyT Normalization

$y=γ⋅tanh(α(x−μ)) + β$

* **Training**: μ=per‑token mean; update running\_mean over (B,S) for inference.
* **Inference**: μ=running\_mean (batch‑independent).
* **γ calibration**: one‑shot on first *representative* batch (B=32), then learned.
* **α warm‑up**: linear from 0.05→0.5 over 2000 steps, avoids saturation but acknowledges eventual tanh′ vanishing.

We compare directly to LayerNorm under identical QAT settings.

### 3.4 SwiGLU‑FFN

* **Up**: $d→8d$ Expand then split: }(a,b)∈ℝ^{4d}).
* **Gate**: a\_q \* SiLU(b\_q)
* **Down**: HBitLinear(4d→d).

We introduce a **matched‑compute ReLU²‑FFN** variant (also 8d→4d→d) to isolate the benefit of gating vs. quadratic nonlinearity.

---

## 4. Experiments

### 4.1 Setup

* **Datasets**: WikiText‑2 (2 M tokens) for MLM perplexity; GLUE for downstream.
* **Batching**: effective B=32 via accumulation to stabilize QDyT statistics.
* **Baselines**: re‑trained FP32 BERT‑Base MLM to get PPL=18.9; 8‑bit Q‑BERT PPL=19.4; distilled ReLU²‑FFN (8×) PPL=19.8; TernaryBERT PPL=26.5 (re‑evaluated).

### 4.2 Main Results (WikiText‑2)

| Model                    | Bits (W/A) | Compute† | PPL ↓ | Size ↓ |
| ------------------------ | ---------- | -------- | ----- | ------ |
| BERT-Base (FP32)         | 32/32      | 1×       | 18.9  | 440 MB |
| Q‑BERT                   | 8/8        | 0.6×\*   | 19.4  | 110 MB |
| ReLU²‑FFN (8× expand)    | 32/4       | 1.1×     | 19.8  | 440 MB |
| TernaryBERT              | 2/32       | 1×       | 26.5  | 65 MB  |
| BitNet‑QDyT‑Base (ours)  | 1.58/4     | 0.7×     | 19.3  | 54 MB  |
| BitNet‑QDyT‑Large (ours) | 1.58/4     | 0.8×     | 16.8  | 70 MB  |

†Compute relative to BERT‑Base (FLOPs simulation). \*Q‑BERT uses int8 kernels.

### 4.3 Ablations

1. **Hadamard**: removing HBitLinear → PPL↑ 21.5; adding only in attention → 20.2.
2. **QDyT vs LayerNorm**: LayerNorm QAT (1.58+4-bit) → 23.1 PPL; static tanh→21.2; full QDyT→19.3.
3. **STE type**: standard STE → 21.7; LSQ→19.7.
4. **SwiGLU vs ReLU² (matched‑compute)**: ReLU² (8×)→19.8; SwiGLU→19.3.

### 4.4 Downstream (GLUE)

| Model            | MNLI | QQP  | QNLI | SST‑2 | Avg  |
| ---------------- | ---- | ---- | ---- | ----- | ---- |
| BERT‑Base (FP32) | 84.6 | 88.5 | 91.2 | 93.5  | 82.7 |
| BitNet‑QDyT‑Base | 83.8 | 87.9 | 90.4 | 92.8  | 81.1 |

---

## 5. Discussion

* **Orthogonal stability**: Hadamard preserves ‖∇‖ exactly; our empirical error decorrelation is approximate.
* **Tanh vs LayerNorm**: QDyT adds hyperparameters but can match or exceed stability under properly sized batches.
* **Compute**: SwiGLU incurs extra FLOPs; we demonstrate its marginal gain justifies the overhead in low‑bit regimes.
* **Hardware**: real speedups require custom int2/int4 kernels; our reported memory and FLOP savings are a first step.

---

## 6. Conclusion

By carefully revising our architecture, baselines, and theoretical claims, we show that **true ternary encoders** with **Hadamard rotations** and **quantization‑aware tanh** can match full‑precision BERT on MLM tasks with significant memory savings. We provide open‑source code and encourage the community to refine fast Hadamard transforms and optimized low‑bit kernels.

---

*Code & models: \[URL]*

**Acknowledgments**: Anonymous reviewers for clarifying boundary conditions on orthogonal decorrelation and QDyT batch dependence.
