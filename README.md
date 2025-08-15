BitNet-QDyT-v2: Orthogonality-Aware Ternary Encoders with Percentile-Scaled Int4 Activations and Mixed-Precision Attention

Authors: Jithin VG

⸻

Abstract

We present BitNet-QDyT-v2, a ternary (1.58-bit) transformer encoder architecture with 4-bit activations that is engineered for both accuracy and deployable efficiency. Building on prior work (Hadamard-augmented ternary layers and a tanh-based normalization, QDyT), we introduce four principled upgrades:
	1.	Orthogonal mixing without dense rotations: we replace dense Hadamards with Block-Hadamard (per-block FHT on power-of-two groups) plus Diagonal-Permutation-Diagonal (DPD) shuffles. This keeps strict orthogonality, improves error decorrelation, and eliminates O($d^2$) overhead.
	2.	Quantization that resists outliers: activations use percentile-based per-channel scaling with EMA and stochastic rounding; weights use TTQ/LSQ+ (learned per-channel scale and threshold, with proper LSQ gradient scaling and clipping).
	3.	Where precision matters, spend it: mixed-precision attention (slightly higher precision for Q/K activations and late-stage V/O or final blocks) and mixed-precision SwiGLU gates (8-bit gate branch) stabilize softmax and gating while preserving ternary/4-bit memory wins.
	4.	Normalization that survives small batches: Group-wise QDyT with a PACT-style learned clip and a post-training calibration pass for running means, plus SkipInit residual scaling for deep stability.

We provide the full architecture, theoretical foundations (norm- and Lipschitz-preserving orthogonal mixing; SQNR analysis for percentile scaling; stability of bounded residual networks under QAT), and practical training algorithms (progressive quantization, QDrop, teacher guidance). We also give best-practice checklists, kernel-friendly layouts, and diagnostics to ensure claimed FLOP/memory advantages translate into wall-clock gains.

⸻

1 Introduction

Quantizing encoder-only transformers to ternary weights and int4 activations is attractive for memory/bandwidth-limited inference and training. However, bidirectional attention, masked-LM objectives, and LayerNorm’s statistic dependence make extreme quantization brittle. BitNet-QDyT-v2 addresses this by (i) enforcing orthogonal, inexpensive channel mixing, (ii) robust scaling against activation outliers, (iii) judicious mixed precision on the most sensitive paths, and (iv) stable normalization under small effective batch sizes.

Our design targets commodity accelerators where speedups require removing dense rotations and aligning bit-layouts with existing int2/int4 kernels.

⸻

2 Background & Notation

Let $x\in\mathbb{R}^{B\times S\times d}$ be the residual stream. A ternary linear maps $xW^\top$, $W\in{-1,0,1}^{d_\text{out}\times d_\text{in}}$ with per-output scale $s_{i}>0$. A $b$-bit symmetric uniform quantizer is $\mathcal{Q}_b(u; \Delta)=\Delta\cdot\mathrm{clip}(\mathrm{round}(u/\Delta),-L,L)$ with $L=2^{b-1}-1$.

Percentile scale. For activations, we estimate $\Delta$ from an EMA of the per-channel $p$-th percentile (e.g., $p\in[99.5,99.9]$) of $|u|$.

Orthogonal mixers. Block-Hadamard: $H=\bigoplus_{k=1}^{K} H_{b}$ with $b\in{32,64,128}$, $Kb=d$. DPD: $O=D_2,P,D_1$, where $D_i$ are diagonal $\pm1$ matrices and $P$ a permutation. Both satisfy $O^\top O=I$.

⸻

3 Model

3.1 Orthogonality-Aware Ternary Layers (HBT: Hadamard/Block + Ternary)

We replace a dense pre-/post-Hadamard with:
	•	Per-head mixing for attention. Apply $O_Q,O_K,O_V$ as head-local block-Hadamard (e.g., $b=64$) or DPD before ternary $W_{QKV}$, and fold one mixer into the projection to avoid extra launches.
	•	FFN mixing. FFN block uses $O_1$ before the up-projection and $O_2$ before the down-projection; we use DPD every layer and Block-H every other block.

This preserves gradient norms and decorrelates channel-wise quantization noise while keeping runtime overhead near zero (bitwise sign-flips and index shuffles are essentially free; block-FHT is $O(d\log b)$).

3.2 Ternary Weights with TTQ/LSQ+

Per output channel $i$:
\[
\hat{w}\_{ij} = s\i \cdot \mathrm{sign}(w\{ij}-t\i)\cdot \mathbf{1}(|w\{ij}-t\_i|>\delta\_i)
\]
We learn $(s_i,t_i,\delta_i)$ with:
	•	Positive scale: $s_i=\mathrm{softplus}(\tilde{s}_i)$.
	•	Thresholds: $t_i$ initialized from the 30th percentile of $|w|$; $\delta_i$ small (e.g., 0.05·median$|w|$) and learned.
	•	LSQ gradient scaling: for scale parameters,
\[
\frac{\partial \mathcal{L}}{\partial s\i} \gets \frac{1}{\sqrt{N\i\cdot L}}\sum\{j}\left(\frac{\partial \mathcal{L}}{\partial \hat{w}\{ij}}\cdot q\{ij}\right),\quad q\{ij}\in\{-1,0,1\},
\]
then clip to $[-1,1]$ before the optimizer step. (Analogous scaling for $t_i,\delta_i$.)

We add uniform dither $u\sim\mathcal{U}(-\tfrac{1}{2}\Delta,\tfrac{1}{2}\Delta)$ during warm-up and ternary dropout (randomly zero 5–10% signs) for 5–10k steps to avoid early dead-zones.

3.3 Activation Quantization with Percentile EMA & Stochastic Rounding

For channel $c$:
	•	Maintain EMA of $p$-th percentile $\rho_c$ of $|x_c|$; set $\Delta_c=\rho_c/L$.
	•	Quantize $x_c$ to int4 with stochastic rounding in training, deterministic at eval.

This allocates codebook levels to the bulk of activations rather than rare spikes.

3.4 QDyT-GN: Group-wise Dynamic Tanh with PACT Clip and Calibration

We replace LayerNorm with:
\mathrm{QDyT}(x) = \gamma \odot \tanh\!\big(\alpha\cdot \mathrm{clip}(x-\mu,\; a)\big) + \beta.
	•	Group-wise $\mu$: mean over small channel groups (e.g., 8–16) per token.
	•	Learned parameters: $\alpha, \gamma, \beta, a$ (with $a=\mathrm{softplus}(\tilde{a})$).
	•	Warm-up: linearly ramp $\alpha$ from $0.05\to0.5$ over the first 2k steps.
	•	Calibration: post-training pass over a held-out shard to recompute running $\mu$ per layer (EMA), then freeze for inference.

3.5 Mixed-Precision Where It Counts
	•	Attention path. Keep Q/K activations at 6–8 bits (per-head scales), accumulate in int32/FP16, apply softmax in FP16, re-quantize outputs. Optionally keep V and $W_O$ weights at 6–8 bits in the last 1–2 blocks.
	•	SwiGLU gate. Gate branch (SiLU) activations at 8 bits; main branch remains int4. We optionally keep gate weights at int4 or int8 (ablate).
	•	Embeddings & LM head. Use 6–8-bit per-row group quant; untie LM head from input embeddings.

3.6 Residual-Path Stability
	•	SkipInit/DeepNet scaling. Residual branch scaled by $\zeta=\frac{0.5}{\sqrt{L}}$ at init (learnable).
	•	μParam. Set learning rates and weight inits so update magnitudes are depth-invariant; practically, scale LR by $1/\sqrt{d}$ for ternary layers and keep optimizer state in FP32.

3.7 Architecture Summary (Base)
	•	24 layers, $d=768$ (12 heads of 64), SwiGLU FFN with 6–8× expand (default 6×, with gate at 8b).
	•	Per-head Block-Hadamard ($b=64$) before Q/K/V; DPD between projections; FFN uses DPD each layer and Block-H in alternating layers.
	•	W: ternary with TTQ/LSQ+; A: int4 except Q/K/gate/late-V/O/embeddings/LM which are 6–8b.

⸻

4 Theoretical Foundations

4.1 Norm Preservation & Error Decorrelation

For an orthogonal $O$, $|Ox|=|x|$; backprop through $O$ preserves gradient norm. Let $\varepsilon$ denote quantization noise with $\mathbb{E}[\varepsilon]=0$. If channels are correlated, per-channel quantizers allocate levels sub-optimally. Applying $O$ yields $\tilde{x}=Ox$ with (approximately) whitened channels; thus the sum of per-channel SQNRs increases (Jensen-type argument). Block-H/DPD approximate this effect with negligible cost.

4.2 Percentile vs Absmax Scaling

Let $X$ be a heavy-tailed activation with tail index $\kappa$. Absmax scaling sets $\Delta=\max |X|/L$, causing the effective dynamic range to be dominated by rare events: utilization $\mathbb{E}[#\text{bins}]$ collapses as $p(|X|>\tau)$ grows. Using the $p$-th percentile $\rho_p$ gives $\Delta=\rho_p/L$, maximizing codebook utilization under a bounded overload probability $1-p$, which upper-bounds mean-squared quantization error by
\[
\mathbb{E}\!\left[(X-\mathcal{Q}\b(X))^2\right] \le \underbrace{\frac{\Delta^2}{12}}\{\text{in-range}} + \underbrace{\mathbb{E}[(|X|-\rho\_p)^2\,\mathbf{1}(|X|>\rho\p)]}\{\text{overload}},
\]
and the second term can be driven down by EMA and moderate $p$ (e.g., 99.5–99.9).

4.3 Bounded Residual Networks with QDyT

With $\phi(x)=\gamma\tanh(\alpha,\mathrm{clip}(x-\mu,a))+\beta$ and $|\phi’(x)|\le |\gamma|\alpha$, each residual block $x\mapsto x + \zeta,f(\phi(x))$ has Lipschitz constant $\le 1 + \zeta L_f |\gamma|\alpha$. With $\zeta\approx O(1/\sqrt{L})$ and small $|\gamma|\alpha$ early in training (warm-up), depth-wise amplification is controlled, yielding stable forward/backward dynamics even with STE.

4.4 Attention Stability Under Mixed Precision

Let $q=\mathcal{Q}_{b_Q}(OxW_Q)$, $k=\mathcal{Q}_{b_K}(OxW_K)$. With per-head asymmetric scales $s_Q,s_K$ and int32 accumulation, the variance of $qk^\top$ is preserved up to $O(\Delta_Q^2+\Delta_K^2)$; softmax logits maintain separability if $\Delta_Q,\Delta_K$ are small (6–8b), explaining the large stability gain from spending bits here.

⸻

5 Training Procedure

5.1 Progressive Quantization
	1.	Steps 0–10%: W ternary off (int8 weights), A at int8, QDyT warm-up.
	2.	Steps 10–40%: Enable ternary in top half of layers; A to int4 except Q/K/gate/emb/LM (6–8b).
	3.	Steps 40–100%: Ternary everywhere (except optional late V/O); keep mixed-precision sites; reduce LR by ×0.7.

5.2 Optimization
	•	Optimizer: AdamW (β₁=0.9, β₂=0.98, ε=1e-8), cosine LR with 10k warm-up.
	•	LSQ+ gradients: clip STE grads in $[-1,1]$ then apply LSQ scaling.
	•	QDrop: bypass weight and/or activation quantizers on 5–10% of mini-batches for the first 20–30% steps.
	•	Teacher guidance (optional): KL on MLM logits from an FP16 teacher or EMA of the model before ternarization (τ=1.5–2.0).

5.3 QDyT Calibration (Post-Training)

Run one pass over a large held-out shard; recompute running $\mu$ per layer (EMA 0.9–0.99) and re-freeze. This is fast and stabilizes inference in deployment.

⸻

6 Algorithms

Algorithm 1: Percentile-EMA Activation Scaling

# per channel c
def update_scale(percentile_tracker, x_c, p=99.5, ema=0.95):
    rho = percentile(|x_c|, p)          # robust to outliers
    percentile_tracker[c] = ema*percentile_tracker[c] + (1-ema)*rho
    Delta_c = percentile_tracker[c] / (2**(b-1)-1)
    return Delta_c

Algorithm 2: TTQ/LSQ+ Ternarization (per output channel i)

# Forward
q_ij = sign(w_ij - t_i) * (abs(w_ij - t_i) > delta_i)  # in {-1,0,1}
w_hat_ij = softplus(s_i) * q_ij

# Backward (pseudo)
grad_s_i = clip(sum_j(grad_w_hat_ij * q_ij) / sqrt(N_i * L), -1, 1)
grad_t_i = ...  # STE + LSQ scaling; similarly clipped
grad_delta_i = ...

Algorithm 3: QDyT-GN with PACT Clip

def QDyT(x, mu, alpha, gamma, beta, a):
    y = x - mu
    y = clip(y, -a, a)                  # PACT-style learned clip
    y = tanh(alpha * y)
    return gamma * y + beta

Algorithm 4: Progressive Quantization & QDrop (high level)

for step, batch in enumerate(loader):
    cfg = schedule(step)                # which layers/paths are low-bit
    with quantization(cfg, qdrop_prob=anneal(step)):
        loss = MLM(model(batch))
        loss += kd_coeff(step) * KL(student_logits, teacher_logits)
    loss.backward()
    clip_gradients(model, 1.0)
    optim.step(); optim.zero_grad()


⸻

7 Implementation Details & Kernel Notes
	•	Block size. Prefer $b=64$ for head-local FHT (fits power-of-two; good cache behavior).
	•	DPD fusing. Implement $D$ as sign-flip bitmasks and $P$ as index remaps; fold into adjacent linear kernels to avoid extra launches.
	•	Accumulators. Use int32 or FP16 accumulation for matmuls; softmax in FP16, re-quantize outputs.
	•	Layouts. Store ternary weights in 2-bit packed format with row-major per-channel scales; group scales with cacheline-aligned strides.
	•	Calibration. Maintain percentile trackers in FP16/FP32; store final scales as FP16.

⸻

8 Evaluation Protocol (Recommended)
	•	Tasks. MLM perplexity on WikiText-2; GLUE downstream (MNLI, QQP, QNLI, SST-2).
	•	Baselines. FP32 BERT-Base; prior BitNet-QDyT; 8-bit baselines (Q-BERT).
	•	Ablations. (a) DPD only vs Block-H+DPD vs dense H; (b) absmax vs percentile; (c) int4 gate vs 8-bit gate; (d) mixed-precision attention on/off; (e) TTQ/LSQ+ vs fixed ternary thresholds; (f) with/without QDyT calibration; (g) expand 6× vs 8×.
	•	Diagnostics. Report per-layer: tanh saturation %, activation codebook utilization, ternary occupancy (% −1/0/+1), Q/K scales, and wall-clock by block.
	•	Reporting. Provide FLOP including mixers, bytes moved, and measured latency/throughput.

⸻

9 Best Practices & Checklists
	•	Use percentile-EMA (99.5–99.9) for activations. Absmax is brittle under int4.
	•	Spend bits wisely. Q/K (6–8b), gate activations (8b), embeddings & LM head (6–8b).
	•	Avoid dense rotations. Prefer Block-H (per head) + DPD; fuse where possible.
	•	Stabilize early. QDrop, dither, and SkipInit prevent early collapse.
	•	Calibrate. One post-train pass for QDyT running means substantially improves inference stability.
	•	Measure what matters. Always include wall-clock and bandwidth, not just simulated FLOPs.

⸻

10 Limitations
	•	Mixed-precision introduces implementation complexity; benefits depend on kernel support.
	•	Percentile estimation adds slight training overhead (mitigated by EMA and shared buffers).
	•	Orthogonal mixing gains depend on how well FHT/DPD are fused into linear kernels.

⸻

11 Conclusion

BitNet-QDyT-v2 replaces expensive dense rotations with Block-Hadamard + DPD, uses robust percentile scaling and TTQ/LSQ+ for low-bit robustness, and spends bits only where they materially improve stability (attention and gates). Together with QDyT-GN, SkipInit, and progressive quantization, the design is both trainable and deployable under 1.58-bit weights / 4-bit activations.

⸻

Appendix A: Reference Hyperparameters
	•	Batching. Effective batch 32–64 via grad accumulation; sequence length 128–256.
	•	LRs. Base LR 3e-4 (ternary), 1.5e-4 (8-bit components), weight decay 0.01.
	•	Warm-up. 10k steps; α ramp 0.05→0.5; kd_coeff cosine 0→0.5.
	•	Percentile EMA. Momentum 0.95 (train), 0.99 (calibration); $p=99.7$ default.

⸻

Appendix B: Suggested Ablation Table Templates

Effect of Orthogonal Mixing (WikiText-2 MLM PPL)

Mixer	Extra cost	PPL ↓	Notes
None	0	—	baseline ternary
DPD	~0		free shuffles
Block-H (per head, b=64) + DPD	low		best accuracy/runtime
Dense H	high		impractical

Effect of Activation Scaling

Scaling	Bits (A)	Utilization ↑	PPL ↓
Absmax	4	low	
Percentile-EMA (99.7) + SR	4	high	

(Fill with measured numbers when you run the suite.)

⸻

Appendix C: Reproducibility Notes
	•	Fix seeds and DPD permutations per run.
	•	Log per-layer codebook usage and tanh saturation.
	•	Export quantizer scales with commit hash; freeze them for eval.

⸻

Appendix D: Minimal PyTorch Snippets (Illustrative)

class DPD(nn.Module):
    def __init__(self, d, perm=None):
        super().__init__()
        self.sign1 = nn.Parameter(torch.ones(d), requires_grad=False)
        self.sign2 = nn.Parameter(torch.ones(d), requires_grad=False)
        self.perm = torch.randperm(d) if perm is None else perm
    def forward(self, x):
        # x: [B,S,d]
        x = x * self.sign1
        x = x[..., self.perm]
        x = x * self.sign2
        return x

def block_hadamard(x, b=64):
    # x: [*, d], d % b == 0
    B = x.shape[:-1]; d = x.shape[-1]; k = d//b
    x = x.reshape(*B, k, b)
    # apply FHT along last dim (use a fused kernel in practice)
    for m in range(int(math.log2(b))):
        stride = 1 << m
        a = x[..., 0::2*stride, :]
        c = x[..., stride::2*stride, :]
        x[..., 0::2*stride, :] = a + c
        x[..., stride::2*stride, :] = a - c
    return x.reshape(*B, d) / math.sqrt(b)

These snippets illustrate interfaces only; production code should use fused, quantized kernels and packed 2-bit weights.

⸻

End of paper.BitNet-QDyT-v2: Orthogonality-Aware Ternary Encoders with Percentile-Scaled Int4 Activations and Mixed-Precision Attention

Authors: Anonymous

⸻

Abstract

We present BitNet-QDyT-v2, a ternary (1.58-bit) transformer encoder architecture with 4-bit activations that is engineered for both accuracy and deployable efficiency. Building on prior work (Hadamard-augmented ternary layers and a tanh-based normalization, QDyT), we introduce four principled upgrades:
	1.	Orthogonal mixing without dense rotations: we replace dense Hadamards with Block-Hadamard (per-block FHT on power-of-two groups) plus Diagonal-Permutation-Diagonal (DPD) shuffles. This keeps strict orthogonality, improves error decorrelation, and eliminates O($d^2$) overhead.
	2.	Quantization that resists outliers: activations use percentile-based per-channel scaling with EMA and stochastic rounding; weights use TTQ/LSQ+ (learned per-channel scale and threshold, with proper LSQ gradient scaling and clipping).
	3.	Where precision matters, spend it: mixed-precision attention (slightly higher precision for Q/K activations and late-stage V/O or final blocks) and mixed-precision SwiGLU gates (8-bit gate branch) stabilize softmax and gating while preserving ternary/4-bit memory wins.
	4.	Normalization that survives small batches: Group-wise QDyT with a PACT-style learned clip and a post-training calibration pass for running means, plus SkipInit residual scaling for deep stability.

We provide the full architecture, theoretical foundations (norm- and Lipschitz-preserving orthogonal mixing; SQNR analysis for percentile scaling; stability of bounded residual networks under QAT), and practical training algorithms (progressive quantization, QDrop, teacher guidance). We also give best-practice checklists, kernel-friendly layouts, and diagnostics to ensure claimed FLOP/memory advantages translate into wall-clock gains.

⸻

1 Introduction

Quantizing encoder-only transformers to ternary weights and int4 activations is attractive for memory/bandwidth-limited inference and training. However, bidirectional attention, masked-LM objectives, and LayerNorm’s statistic dependence make extreme quantization brittle. BitNet-QDyT-v2 addresses this by (i) enforcing orthogonal, inexpensive channel mixing, (ii) robust scaling against activation outliers, (iii) judicious mixed precision on the most sensitive paths, and (iv) stable normalization under small effective batch sizes.

Our design targets commodity accelerators where speedups require removing dense rotations and aligning bit-layouts with existing int2/int4 kernels.

⸻

2 Background & Notation

Let $x\in\mathbb{R}^{B\times S\times d}$ be the residual stream. A ternary linear maps $xW^\top$, $W\in{-1,0,1}^{d_\text{out}\times d_\text{in}}$ with per-output scale $s_{i}>0$. A $b$-bit symmetric uniform quantizer is $\mathcal{Q}_b(u; \Delta)=\Delta\cdot\mathrm{clip}(\mathrm{round}(u/\Delta),-L,L)$ with $L=2^{b-1}-1$.

Percentile scale. For activations, we estimate $\Delta$ from an EMA of the per-channel $p$-th percentile (e.g., $p\in[99.5,99.9]$) of $|u|$.

Orthogonal mixers. Block-Hadamard: $H=\bigoplus_{k=1}^{K} H_{b}$ with $b\in{32,64,128}$, $Kb=d$. DPD: $O=D_2,P,D_1$, where $D_i$ are diagonal $\pm1$ matrices and $P$ a permutation. Both satisfy $O^\top O=I$.

⸻

3 Model

3.1 Orthogonality-Aware Ternary Layers (HBT: Hadamard/Block + Ternary)

We replace a dense pre-/post-Hadamard with:
	•	Per-head mixing for attention. Apply $O_Q,O_K,O_V$ as head-local block-Hadamard (e.g., $b=64$) or DPD before ternary $W_{QKV}$, and fold one mixer into the projection to avoid extra launches.
	•	FFN mixing. FFN block uses $O_1$ before the up-projection and $O_2$ before the down-projection; we use DPD every layer and Block-H every other block.

This preserves gradient norms and decorrelates channel-wise quantization noise while keeping runtime overhead near zero (bitwise sign-flips and index shuffles are essentially free; block-FHT is $O(d\log b)$).

3.2 Ternary Weights with TTQ/LSQ+

Per output channel $i$:
\[
\hat{w}\_{ij} = s\i \cdot \mathrm{sign}(w\{ij}-t\i)\cdot \mathbf{1}(|w\{ij}-t\_i|>\delta\_i)
\]
We learn $(s_i,t_i,\delta_i)$ with:
	•	Positive scale: $s_i=\mathrm{softplus}(\tilde{s}_i)$.
	•	Thresholds: $t_i$ initialized from the 30th percentile of $|w|$; $\delta_i$ small (e.g., 0.05·median$|w|$) and learned.
	•	LSQ gradient scaling: for scale parameters,
\[
\frac{\partial \mathcal{L}}{\partial s\i} \gets \frac{1}{\sqrt{N\i\cdot L}}\sum\{j}\left(\frac{\partial \mathcal{L}}{\partial \hat{w}\{ij}}\cdot q\{ij}\right),\quad q\{ij}\in\{-1,0,1\},
\]
then clip to $[-1,1]$ before the optimizer step. (Analogous scaling for $t_i,\delta_i$.)

We add uniform dither $u\sim\mathcal{U}(-\tfrac{1}{2}\Delta,\tfrac{1}{2}\Delta)$ during warm-up and ternary dropout (randomly zero 5–10% signs) for 5–10k steps to avoid early dead-zones.

3.3 Activation Quantization with Percentile EMA & Stochastic Rounding

For channel $c$:
	•	Maintain EMA of $p$-th percentile $\rho_c$ of $|x_c|$; set $\Delta_c=\rho_c/L$.
	•	Quantize $x_c$ to int4 with stochastic rounding in training, deterministic at eval.

This allocates codebook levels to the bulk of activations rather than rare spikes.

3.4 QDyT-GN: Group-wise Dynamic Tanh with PACT Clip and Calibration

We replace LayerNorm with:
\mathrm{QDyT}(x) = \gamma \odot \tanh\!\big(\alpha\cdot \mathrm{clip}(x-\mu,\; a)\big) + \beta.
	•	Group-wise $\mu$: mean over small channel groups (e.g., 8–16) per token.
	•	Learned parameters: $\alpha, \gamma, \beta, a$ (with $a=\mathrm{softplus}(\tilde{a})$).
	•	Warm-up: linearly ramp $\alpha$ from $0.05\to0.5$ over the first 2k steps.
	•	Calibration: post-training pass over a held-out shard to recompute running $\mu$ per layer (EMA), then freeze for inference.

3.5 Mixed-Precision Where It Counts
	•	Attention path. Keep Q/K activations at 6–8 bits (per-head scales), accumulate in int32/FP16, apply softmax in FP16, re-quantize outputs. Optionally keep V and $W_O$ weights at 6–8 bits in the last 1–2 blocks.
	•	SwiGLU gate. Gate branch (SiLU) activations at 8 bits; main branch remains int4. We optionally keep gate weights at int4 or int8 (ablate).
	•	Embeddings & LM head. Use 6–8-bit per-row group quant; untie LM head from input embeddings.

3.6 Residual-Path Stability
	•	SkipInit/DeepNet scaling. Residual branch scaled by $\zeta=\frac{0.5}{\sqrt{L}}$ at init (learnable).
	•	μParam. Set learning rates and weight inits so update magnitudes are depth-invariant; practically, scale LR by $1/\sqrt{d}$ for ternary layers and keep optimizer state in FP32.

3.7 Architecture Summary (Base)
	•	24 layers, $d=768$ (12 heads of 64), SwiGLU FFN with 6–8× expand (default 6×, with gate at 8b).
	•	Per-head Block-Hadamard ($b=64$) before Q/K/V; DPD between projections; FFN uses DPD each layer and Block-H in alternating layers.
	•	W: ternary with TTQ/LSQ+; A: int4 except Q/K/gate/late-V/O/embeddings/LM which are 6–8b.

⸻

4 Theoretical Foundations

4.1 Norm Preservation & Error Decorrelation

For an orthogonal $O$, $|Ox|=|x|$; backprop through $O$ preserves gradient norm. Let $\varepsilon$ denote quantization noise with $\mathbb{E}[\varepsilon]=0$. If channels are correlated, per-channel quantizers allocate levels sub-optimally. Applying $O$ yields $\tilde{x}=Ox$ with (approximately) whitened channels; thus the sum of per-channel SQNRs increases (Jensen-type argument). Block-H/DPD approximate this effect with negligible cost.

4.2 Percentile vs Absmax Scaling

Let $X$ be a heavy-tailed activation with tail index $\kappa$. Absmax scaling sets $\Delta=\max |X|/L$, causing the effective dynamic range to be dominated by rare events: utilization $\mathbb{E}[#\text{bins}]$ collapses as $p(|X|>\tau)$ grows. Using the $p$-th percentile $\rho_p$ gives $\Delta=\rho_p/L$, maximizing codebook utilization under a bounded overload probability $1-p$, which upper-bounds mean-squared quantization error by
\[
\mathbb{E}\!\left[(X-\mathcal{Q}\b(X))^2\right] \le \underbrace{\frac{\Delta^2}{12}}\{\text{in-range}} + \underbrace{\mathbb{E}[(|X|-\rho\_p)^2\,\mathbf{1}(|X|>\rho\p)]}\{\text{overload}},
\]
and the second term can be driven down by EMA and moderate $p$ (e.g., 99.5–99.9).

4.3 Bounded Residual Networks with QDyT

With $\phi(x)=\gamma\tanh(\alpha,\mathrm{clip}(x-\mu,a))+\beta$ and $|\phi’(x)|\le |\gamma|\alpha$, each residual block $x\mapsto x + \zeta,f(\phi(x))$ has Lipschitz constant $\le 1 + \zeta L_f |\gamma|\alpha$. With $\zeta\approx O(1/\sqrt{L})$ and small $|\gamma|\alpha$ early in training (warm-up), depth-wise amplification is controlled, yielding stable forward/backward dynamics even with STE.

4.4 Attention Stability Under Mixed Precision

Let $q=\mathcal{Q}_{b_Q}(OxW_Q)$, $k=\mathcal{Q}_{b_K}(OxW_K)$. With per-head asymmetric scales $s_Q,s_K$ and int32 accumulation, the variance of $qk^\top$ is preserved up to $O(\Delta_Q^2+\Delta_K^2)$; softmax logits maintain separability if $\Delta_Q,\Delta_K$ are small (6–8b), explaining the large stability gain from spending bits here.

⸻

5 Training Procedure

5.1 Progressive Quantization
	1.	Steps 0–10%: W ternary off (int8 weights), A at int8, QDyT warm-up.
	2.	Steps 10–40%: Enable ternary in top half of layers; A to int4 except Q/K/gate/emb/LM (6–8b).
	3.	Steps 40–100%: Ternary everywhere (except optional late V/O); keep mixed-precision sites; reduce LR by ×0.7.

5.2 Optimization
	•	Optimizer: AdamW (β₁=0.9, β₂=0.98, ε=1e-8), cosine LR with 10k warm-up.
	•	LSQ+ gradients: clip STE grads in $[-1,1]$ then apply LSQ scaling.
	•	QDrop: bypass weight and/or activation quantizers on 5–10% of mini-batches for the first 20–30% steps.
	•	Teacher guidance (optional): KL on MLM logits from an FP16 teacher or EMA of the model before ternarization (τ=1.5–2.0).

5.3 QDyT Calibration (Post-Training)

Run one pass over a large held-out shard; recompute running $\mu$ per layer (EMA 0.9–0.99) and re-freeze. This is fast and stabilizes inference in deployment.

⸻

6 Algorithms

Algorithm 1: Percentile-EMA Activation Scaling

# per channel c
def update_scale(percentile_tracker, x_c, p=99.5, ema=0.95):
    rho = percentile(|x_c|, p)          # robust to outliers
    percentile_tracker[c] = ema*percentile_tracker[c] + (1-ema)*rho
    Delta_c = percentile_tracker[c] / (2**(b-1)-1)
    return Delta_c

Algorithm 2: TTQ/LSQ+ Ternarization (per output channel i)

# Forward
q_ij = sign(w_ij - t_i) * (abs(w_ij - t_i) > delta_i)  # in {-1,0,1}
w_hat_ij = softplus(s_i) * q_ij

# Backward (pseudo)
grad_s_i = clip(sum_j(grad_w_hat_ij * q_ij) / sqrt(N_i * L), -1, 1)
grad_t_i = ...  # STE + LSQ scaling; similarly clipped
grad_delta_i = ...

Algorithm 3: QDyT-GN with PACT Clip

def QDyT(x, mu, alpha, gamma, beta, a):
    y = x - mu
    y = clip(y, -a, a)                  # PACT-style learned clip
    y = tanh(alpha * y)
    return gamma * y + beta

Algorithm 4: Progressive Quantization & QDrop (high level)

for step, batch in enumerate(loader):
    cfg = schedule(step)                # which layers/paths are low-bit
    with quantization(cfg, qdrop_prob=anneal(step)):
        loss = MLM(model(batch))
        loss += kd_coeff(step) * KL(student_logits, teacher_logits)
    loss.backward()
    clip_gradients(model, 1.0)
    optim.step(); optim.zero_grad()


⸻

7 Implementation Details & Kernel Notes
	•	Block size. Prefer $b=64$ for head-local FHT (fits power-of-two; good cache behavior).
	•	DPD fusing. Implement $D$ as sign-flip bitmasks and $P$ as index remaps; fold into adjacent linear kernels to avoid extra launches.
	•	Accumulators. Use int32 or FP16 accumulation for matmuls; softmax in FP16, re-quantize outputs.
	•	Layouts. Store ternary weights in 2-bit packed format with row-major per-channel scales; group scales with cacheline-aligned strides.
	•	Calibration. Maintain percentile trackers in FP16/FP32; store final scales as FP16.

⸻

8 Evaluation Protocol (Recommended)
	•	Tasks. MLM perplexity on WikiText-2; GLUE downstream (MNLI, QQP, QNLI, SST-2).
	•	Baselines. FP32 BERT-Base; prior BitNet-QDyT; 8-bit baselines (Q-BERT).
	•	Ablations. (a) DPD only vs Block-H+DPD vs dense H; (b) absmax vs percentile; (c) int4 gate vs 8-bit gate; (d) mixed-precision attention on/off; (e) TTQ/LSQ+ vs fixed ternary thresholds; (f) with/without QDyT calibration; (g) expand 6× vs 8×.
	•	Diagnostics. Report per-layer: tanh saturation %, activation codebook utilization, ternary occupancy (% −1/0/+1), Q/K scales, and wall-clock by block.
	•	Reporting. Provide FLOP including mixers, bytes moved, and measured latency/throughput.

⸻

9 Best Practices & Checklists
	•	Use percentile-EMA (99.5–99.9) for activations. Absmax is brittle under int4.
	•	Spend bits wisely. Q/K (6–8b), gate activations (8b), embeddings & LM head (6–8b).
	•	Avoid dense rotations. Prefer Block-H (per head) + DPD; fuse where possible.
	•	Stabilize early. QDrop, dither, and SkipInit prevent early collapse.
	•	Calibrate. One post-train pass for QDyT running means substantially improves inference stability.
	•	Measure what matters. Always include wall-clock and bandwidth, not just simulated FLOPs.

⸻

10 Limitations
	•	Mixed-precision introduces implementation complexity; benefits depend on kernel support.
	•	Percentile estimation adds slight training overhead (mitigated by EMA and shared buffers).
	•	Orthogonal mixing gains depend on how well FHT/DPD are fused into linear kernels.

⸻

11 Conclusion

BitNet-QDyT-v2 replaces expensive dense rotations with Block-Hadamard + DPD, uses robust percentile scaling and TTQ/LSQ+ for low-bit robustness, and spends bits only where they materially improve stability (attention and gates). Together with QDyT-GN, SkipInit, and progressive quantization, the design is both trainable and deployable under 1.58-bit weights / 4-bit activations.

⸻

Appendix A: Reference Hyperparameters
	•	Batching. Effective batch 32–64 via grad accumulation; sequence length 128–256.
	•	LRs. Base LR 3e-4 (ternary), 1.5e-4 (8-bit components), weight decay 0.01.
	•	Warm-up. 10k steps; α ramp 0.05→0.5; kd_coeff cosine 0→0.5.
	•	Percentile EMA. Momentum 0.95 (train), 0.99 (calibration); $p=99.7$ default.

⸻

Appendix B: Suggested Ablation Table Templates

Effect of Orthogonal Mixing (WikiText-2 MLM PPL)

Mixer	Extra cost	PPL ↓	Notes
None	0	—	baseline ternary
DPD	~0		free shuffles
Block-H (per head, b=64) + DPD	low		best accuracy/runtime
Dense H	high		impractical

Effect of Activation Scaling

Scaling	Bits (A)	Utilization ↑	PPL ↓
Absmax	4	low	
Percentile-EMA (99.7) + SR	4	high	

(Fill with measured numbers when you run the suite.)

⸻

Appendix C: Reproducibility Notes
	•	Fix seeds and DPD permutations per run.
	•	Log per-layer codebook usage and tanh saturation.
	•	Export quantizer scales with commit hash; freeze them for eval.

⸻

Appendix D: Minimal PyTorch Snippets (Illustrative)

class DPD(nn.Module):
    def __init__(self, d, perm=None):
        super().__init__()
        self.sign1 = nn.Parameter(torch.ones(d), requires_grad=False)
        self.sign2 = nn.Parameter(torch.ones(d), requires_grad=False)
        self.perm = torch.randperm(d) if perm is None else perm
    def forward(self, x):
        # x: [B,S,d]
        x = x * self.sign1
        x = x[..., self.perm]
        x = x * self.sign2
        return x

def block_hadamard(x, b=64):
    # x: [*, d], d % b == 0
    B = x.shape[:-1]; d = x.shape[-1]; k = d//b
    x = x.reshape(*B, k, b)
    # apply FHT along last dim (use a fused kernel in practice)
    for m in range(int(math.log2(b))):
        stride = 1 << m
        a = x[..., 0::2*stride, :]
        c = x[..., stride::2*stride, :]
        x[..., 0::2*stride, :] = a + c
        x[..., stride::2*stride, :] = a - c
    return x.reshape(*B, d) / math.sqrt(b)

These snippets illustrate interfaces only; production code should use fused, quantized kernels and packed 2-bit weights.

⸻

End of paper.
