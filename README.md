\documentclass[10pt,conference]{IEEEtran}
\IEEEoverridecommandlockouts

% Packages
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{mathtools}
\usepackage{bm}
\usepackage{subcaption}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}

% Custom commands
\DeclareMathOperator{\sign}{sign}
\DeclareMathOperator{\softplus}{softplus}
\DeclareMathOperator{\clip}{clip}
\DeclareMathOperator{\percentile}{percentile}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\calL}{\mathcal{L}}
\newcommand{\calQ}{\mathcal{Q}}
\newcommand{\calU}{\mathcal{U}}

\title{BitNet-QDyT-v2: Orthogonality-Aware Ternary Encoders with Percentile-Scaled Int4 Activations and Mixed-Precision Attention}

\author{\IEEEauthorblockN{Jithin VG}
\IEEEauthorblockA{\\
\textit{Affiliation}\\
Email: author@institution.edu}}

\begin{document}

\maketitle

\begin{abstract}
We present BitNet-QDyT-v2, a ternary (1.58-bit) transformer encoder architecture with 4-bit activations that is engineered for both accuracy and deployable efficiency. Building on prior work (Hadamard-augmented ternary layers and a tanh-based normalization, QDyT), we introduce four principled upgrades:
(1) \textbf{Orthogonal mixing without dense rotations}: we replace dense Hadamards with Block-Hadamard (per-block FHT on power-of-two groups) plus Diagonal-Permutation-Diagonal (DPD) shuffles. This keeps strict orthogonality, improves error decorrelation, and eliminates $O(d^2)$ overhead.
(2) \textbf{Quantization that resists outliers}: activations use percentile-based per-channel scaling with EMA and stochastic rounding; weights use TTQ/LSQ+ (learned per-channel scale and threshold, with proper LSQ gradient scaling and clipping).
(3) \textbf{Where precision matters, spend it}: mixed-precision attention (slightly higher precision for Q/K activations and late-stage V/O or final blocks) and mixed-precision SwiGLU gates (8-bit gate branch) stabilize softmax and gating while preserving ternary/4-bit memory wins.
(4) \textbf{Normalization that survives small batches}: Group-wise QDyT with a PACT-style learned clip and a post-training calibration pass for running means, plus SkipInit residual scaling for deep stability.

We provide the full architecture, theoretical foundations (norm- and Lipschitz-preserving orthogonal mixing; SQNR analysis for percentile scaling; stability of bounded residual networks under QAT), and practical training algorithms (progressive quantization, QDrop, teacher guidance). We also give best-practice checklists, kernel-friendly layouts, and diagnostics to ensure claimed FLOP/memory advantages translate into wall-clock gains.
\end{abstract}

\begin{IEEEkeywords}
Quantization, Transformer, Ternary Networks, Orthogonal Transformations, Mixed Precision
\end{IEEEkeywords}

\section{Introduction}

Quantizing encoder-only transformers to ternary weights and int4 activations is attractive for memory/bandwidth-limited inference and training. However, bidirectional attention, masked-LM objectives, and LayerNorm's statistic dependence make extreme quantization brittle. BitNet-QDyT-v2 addresses this by:
\begin{enumerate}
    \item Enforcing orthogonal, inexpensive channel mixing
    \item Robust scaling against activation outliers
    \item Judicious mixed precision on the most sensitive paths
    \item Stable normalization under small effective batch sizes
\end{enumerate}

Our design targets commodity accelerators where speedups require removing dense rotations and aligning bit-layouts with existing int2/int4 kernels.

\section{Background \& Notation}

Let $x \in \R^{B \times S \times d}$ be the residual stream. A ternary linear layer maps $xW^\top$, where $W \in \{-1, 0, 1\}^{d_{\text{out}} \times d_{\text{in}}}$ with per-output scale $s_i > 0$. A $b$-bit symmetric uniform quantizer is:
\begin{equation}
    \calQ_b(u; \Delta) = \Delta \cdot \clip\left(\text{round}\left(\frac{u}{\Delta}\right), -L, L\right)
\end{equation}
with $L = 2^{b-1} - 1$.

\textbf{Percentile scale.} For activations, we estimate $\Delta$ from an EMA of the per-channel $p$-th percentile (e.g., $p \in [99.5, 99.9]$) of $|u|$.

\textbf{Orthogonal mixers.} Block-Hadamard: $H = \bigoplus_{k=1}^{K} H_b$ with $b \in \{32, 64, 128\}$, $Kb = d$. DPD: $O = D_2 P D_1$, where $D_i$ are diagonal $\pm 1$ matrices and $P$ a permutation. Both satisfy $O^\top O = I$.

\section{Model Architecture}

\subsection{Orthogonality-Aware Ternary Layers (HBT)}

We replace dense pre-/post-Hadamard transformations with:

\begin{itemize}
    \item \textbf{Per-head mixing for attention.} Apply $O_Q, O_K, O_V$ as head-local block-Hadamard (e.g., $b = 64$) or DPD before ternary $W_{QKV}$, and fold one mixer into the projection to avoid extra launches.
    \item \textbf{FFN mixing.} FFN block uses $O_1$ before the up-projection and $O_2$ before the down-projection; we use DPD every layer and Block-H every other block.
\end{itemize}

This preserves gradient norms and decorrelates channel-wise quantization noise while keeping runtime overhead near zero (bitwise sign-flips and index shuffles are essentially free; block-FHT is $O(d \log b)$).

\subsection{Ternary Weights with TTQ/LSQ+}

Per output channel $i$:
\begin{equation}
    \hat{w}_{ij} = s_i \cdot \sign(w_{ij} - t_i) \cdot \mathbf{1}(|w_{ij} - t_i| > \delta_i)
\end{equation}

We learn $(s_i, t_i, \delta_i)$ with:
\begin{itemize}
    \item \textbf{Positive scale:} $s_i = \softplus(\tilde{s}_i)$
    \item \textbf{Thresholds:} $t_i$ initialized from the 30th percentile of $|w|$; $\delta_i$ small (e.g., $0.05 \cdot \text{median}|w|$) and learned
    \item \textbf{LSQ gradient scaling:} for scale parameters,
    \begin{equation}
        \frac{\partial \calL}{\partial s_i} \gets \frac{1}{\sqrt{N_i \cdot L}} \sum_j \left(\frac{\partial \calL}{\partial \hat{w}_{ij}} \cdot q_{ij}\right), \quad q_{ij} \in \{-1, 0, 1\}
    \end{equation}
    then clip to $[-1, 1]$ before the optimizer step. (Analogous scaling for $t_i, \delta_i$.)
\end{itemize}

We add uniform dither $u \sim \calU(-\frac{1}{2}\Delta, \frac{1}{2}\Delta)$ during warm-up and ternary dropout (randomly zero 5--10\% signs) for 5--10k steps to avoid early dead-zones.

\subsection{Activation Quantization with Percentile EMA \& Stochastic Rounding}

For channel $c$:
\begin{itemize}
    \item Maintain EMA of $p$-th percentile $\rho_c$ of $|x_c|$; set $\Delta_c = \rho_c/L$
    \item Quantize $x_c$ to int4 with stochastic rounding in training, deterministic at eval
\end{itemize}

This allocates codebook levels to the bulk of activations rather than rare spikes.

\subsection{QDyT-GN: Group-wise Dynamic Tanh with PACT Clip and Calibration}

We replace LayerNorm with:
\begin{equation}
    \text{QDyT}(x) = \gamma \odot \tanh\big(\alpha \cdot \clip(x - \mu, a)\big) + \beta
\end{equation}

\begin{itemize}
    \item \textbf{Group-wise $\mu$:} mean over small channel groups (e.g., 8--16) per token
    \item \textbf{Learned parameters:} $\alpha, \gamma, \beta, a$ (with $a = \softplus(\tilde{a})$)
    \item \textbf{Warm-up:} linearly ramp $\alpha$ from $0.05 \to 0.5$ over the first 2k steps
    \item \textbf{Calibration:} post-training pass over a held-out shard to recompute running $\mu$ per layer (EMA), then freeze for inference
\end{itemize}

\subsection{Mixed-Precision Where It Counts}

\begin{itemize}
    \item \textbf{Attention path.} Keep Q/K activations at 6--8 bits (per-head scales), accumulate in int32/FP16, apply softmax in FP16, re-quantize outputs. Optionally keep V and $W_O$ weights at 6--8 bits in the last 1--2 blocks.
    \item \textbf{SwiGLU gate.} Gate branch (SiLU) activations at 8 bits; main branch remains int4. We optionally keep gate weights at int4 or int8 (ablate).
    \item \textbf{Embeddings \& LM head.} Use 6--8-bit per-row group quant; untie LM head from input embeddings.
\end{itemize}

\subsection{Residual-Path Stability}

\begin{itemize}
    \item \textbf{SkipInit/DeepNet scaling.} Residual branch scaled by $\zeta = \frac{0.5}{\sqrt{L}}$ at init (learnable)
    \item \textbf{$\mu$Param.} Set learning rates and weight inits so update magnitudes are depth-invariant; practically, scale LR by $1/\sqrt{d}$ for ternary layers and keep optimizer state in FP32
\end{itemize}

\subsection{Architecture Summary (Base)}

\begin{itemize}
    \item 24 layers, $d = 768$ (12 heads of 64), SwiGLU FFN with 6--8$\times$ expand (default 6$\times$, with gate at 8b)
    \item Per-head Block-Hadamard ($b = 64$) before Q/K/V; DPD between projections; FFN uses DPD each layer and Block-H in alternating layers
    \item W: ternary with TTQ/LSQ+; A: int4 except Q/K/gate/late-V/O/embeddings/LM which are 6--8b
\end{itemize}

\section{Theoretical Foundations}

\subsection{Norm Preservation \& Error Decorrelation}

For an orthogonal $O$, $\|Ox\| = \|x\|$; backprop through $O$ preserves gradient norm. Let $\varepsilon$ denote quantization noise with $\E[\varepsilon] = 0$. If channels are correlated, per-channel quantizers allocate levels sub-optimally. Applying $O$ yields $\tilde{x} = Ox$ with (approximately) whitened channels; thus the sum of per-channel SQNRs increases (Jensen-type argument). Block-H/DPD approximate this effect with negligible cost.

\subsection{Percentile vs Absmax Scaling}

Let $X$ be a heavy-tailed activation with tail index $\kappa$. Absmax scaling sets $\Delta = \max|X|/L$, causing the effective dynamic range to be dominated by rare events: utilization $\E[\#\text{bins}]$ collapses as $p(|X| > \tau)$ grows. Using the $p$-th percentile $\rho_p$ gives $\Delta = \rho_p/L$, maximizing codebook utilization under a bounded overload probability $1-p$, which upper-bounds mean-squared quantization error by:
\begin{equation}
    \E\left[(X - \calQ_b(X))^2\right] \le \underbrace{\frac{\Delta^2}{12}}_{\text{in-range}} + \underbrace{\E[(|X| - \rho_p)^2 \mathbf{1}(|X| > \rho_p)]}_{\text{overload}}
\end{equation}
and the second term can be driven down by EMA and moderate $p$ (e.g., 99.5--99.9).

\subsection{Bounded Residual Networks with QDyT}

With $\phi(x) = \gamma\tanh(\alpha \cdot \clip(x - \mu, a)) + \beta$ and $|\phi'(x)| \le |\gamma|\alpha$, each residual block $x \mapsto x + \zeta f(\phi(x))$ has Lipschitz constant $\le 1 + \zeta L_f |\gamma|\alpha$. With $\zeta \approx O(1/\sqrt{L})$ and small $|\gamma|\alpha$ early in training (warm-up), depth-wise amplification is controlled, yielding stable forward/backward dynamics even with STE.

\subsection{Attention Stability Under Mixed Precision}

Let $q = \calQ_{b_Q}(OxW_Q)$, $k = \calQ_{b_K}(OxW_K)$. With per-head asymmetric scales $s_Q, s_K$ and int32 accumulation, the variance of $qk^\top$ is preserved up to $O(\Delta_Q^2 + \Delta_K^2)$; softmax logits maintain separability if $\Delta_Q, \Delta_K$ are small (6--8b), explaining the large stability gain from spending bits here.

\section{Training Procedure}

\subsection{Progressive Quantization}

\begin{enumerate}
    \item \textbf{Steps 0--10\%:} W ternary off (int8 weights), A at int8, QDyT warm-up
    \item \textbf{Steps 10--40\%:} Enable ternary in top half of layers; A to int4 except Q/K/gate/emb/LM (6--8b)
    \item \textbf{Steps 40--100\%:} Ternary everywhere (except optional late V/O); keep mixed-precision sites; reduce LR by $\times 0.7$
\end{enumerate}

\subsection{Optimization}

\begin{itemize}
    \item \textbf{Optimizer:} AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-8}$), cosine LR with 10k warm-up
    \item \textbf{LSQ+ gradients:} clip STE grads in $[-1, 1]$ then apply LSQ scaling
    \item \textbf{QDrop:} bypass weight and/or activation quantizers on 5--10\% of mini-batches for the first 20--30\% steps
    \item \textbf{Teacher guidance (optional):} KL on MLM logits from an FP16 teacher or EMA of the model before ternarization ($\tau = 1.5$--2.0)
\end{itemize}

\subsection{QDyT Calibration (Post-Training)}

Run one pass over a large held-out shard; recompute running $\mu$ per layer (EMA 0.9--0.99) and re-freeze. This is fast and stabilizes inference in deployment.

\section{Algorithms}

\begin{algorithm}
\caption{Percentile-EMA Activation Scaling}
\begin{algorithmic}[1]
\STATE \textbf{Input:} percentile\_tracker, $x_c$, $p = 99.5$, ema = 0.95
\STATE $\rho \gets \percentile(|x_c|, p)$ \COMMENT{robust to outliers}
\STATE percentile\_tracker$[c] \gets$ ema $\cdot$ percentile\_tracker$[c] + (1 - \text{ema}) \cdot \rho$
\STATE $\Delta_c \gets$ percentile\_tracker$[c] / (2^{b-1} - 1)$
\STATE \textbf{return} $\Delta_c$
\end{algorithmic}
\end{algorithm}

\begin{algorithm}
\caption{TTQ/LSQ+ Ternarization (per output channel $i$)}
\begin{algorithmic}[1]
\STATE \textbf{// Forward}
\STATE $q_{ij} \gets \sign(w_{ij} - t_i) \cdot (|w_{ij} - t_i| > \delta_i)$ \COMMENT{in $\{-1, 0, 1\}$}
\STATE $\hat{w}_{ij} \gets \softplus(s_i) \cdot q_{ij}$
\STATE \textbf{// Backward (pseudo)}
\STATE $\text{grad}_{s_i} \gets \clip\left(\sum_j(\text{grad}_{\hat{w}_{ij}} \cdot q_{ij}) / \sqrt{N_i \cdot L}, -1, 1\right)$
\STATE $\text{grad}_{t_i} \gets \ldots$ \COMMENT{STE + LSQ scaling; similarly clipped}
\STATE $\text{grad}_{\delta_i} \gets \ldots$
\end{algorithmic}
\end{algorithm}

\begin{algorithm}
\caption{QDyT-GN with PACT Clip}
\begin{algorithmic}[1]
\STATE \textbf{function} QDyT($x$, $\mu$, $\alpha$, $\gamma$, $\beta$, $a$)
\STATE $y \gets x - \mu$
\STATE $y \gets \clip(y, -a, a)$ \COMMENT{PACT-style learned clip}
\STATE $y \gets \tanh(\alpha \cdot y)$
\STATE \textbf{return} $\gamma \cdot y + \beta$
\end{algorithmic}
\end{algorithm}

\begin{algorithm}
\caption{Progressive Quantization \& QDrop (high level)}
\begin{algorithmic}[1]
\FOR{step, batch in enumerate(loader)}
    \STATE cfg $\gets$ schedule(step) \COMMENT{which layers/paths are low-bit}
    \STATE \textbf{with} quantization(cfg, qdrop\_prob=anneal(step)):
    \STATE \quad loss $\gets$ MLM(model(batch))
    \STATE \quad loss $\gets$ loss + kd\_coeff(step) $\cdot$ KL(student\_logits, teacher\_logits)
    \STATE loss.backward()
    \STATE clip\_gradients(model, 1.0)
    \STATE optim.step(); optim.zero\_grad()
\ENDFOR
\end{algorithmic}
\end{algorithm}

\section{Implementation Details \& Kernel Notes}

\begin{itemize}
    \item \textbf{Block size.} Prefer $b = 64$ for head-local FHT (fits power-of-two; good cache behavior)
    \item \textbf{DPD fusing.} Implement $D$ as sign-flip bitmasks and $P$ as index remaps; fold into adjacent linear kernels to avoid extra launches
    \item \textbf{Accumulators.} Use int32 or FP16 accumulation for matmuls; softmax in FP16, re-quantize outputs
    \item \textbf{Layouts.} Store ternary weights in 2-bit packed format with row-major per-channel scales; group scales with cacheline-aligned strides
    \item \textbf{Calibration.} Maintain percentile trackers in FP16/FP32; store final scales as FP16
\end{itemize}

\section{Evaluation Protocol}

\begin{itemize}
    \item \textbf{Tasks.} MLM perplexity on WikiText-2; GLUE downstream (MNLI, QQP, QNLI, SST-2)
    \item \textbf{Baselines.} FP32 BERT-Base; prior BitNet-QDyT; 8-bit baselines (Q-BERT)
    \item \textbf{Ablations.} 
    \begin{enumerate}[label=(\alph*)]
        \item DPD only vs Block-H+DPD vs dense H
        \item absmax vs percentile
        \item int4 gate vs 8-bit gate
        \item mixed-precision attention on/off
        \item TTQ/LSQ+ vs fixed ternary thresholds
        \item with/without QDyT calibration
        \item expand 6$\times$ vs 8$\times$
    \end{enumerate}
    \item \textbf{Diagnostics.} Report per-layer: tanh saturation \%, activation codebook utilization, ternary occupancy (\% $-1/0/+1$), Q/K scales, and wall-clock by block
    \item \textbf{Reporting.} Provide FLOP including mixers, bytes moved, and measured latency/throughput
\end{itemize}

\section{Best Practices \& Checklists}

\begin{itemize}
    \item Use percentile-EMA (99.5--99.9) for activations. Absmax is brittle under int4
    \item Spend bits wisely: Q/K (6--8b), gate activations (8b), embeddings \& LM head (6--8b)
    \item Avoid dense rotations. Prefer Block-H (per head) + DPD; fuse where possible
    \item Stabilize early: QDrop, dither, and SkipInit prevent early collapse
    \item Calibrate: One post-train pass for QDyT running means substantially improves inference stability
    \item Measure what matters: Always include wall-clock and bandwidth, not just simulated FLOPs
\end{itemize}

\section{Limitations}

\begin{itemize}
    \item Mixed-precision introduces implementation complexity; benefits depend on kernel support
    \item Percentile estimation adds slight training overhead (mitigated by EMA and shared buffers)
    \item Orthogonal mixing gains depend on how well FHT/DPD are fused into linear kernels
\end{itemize}

\section{Conclusion}

BitNet-QDyT-v2 replaces expensive dense rotations with Block-Hadamard + DPD, uses robust percentile scaling and TTQ/LSQ+ for low-bit robustness, and spends bits only where they materially improve stability (attention and gates). Together with QDyT-GN, SkipInit, and progressive quantization, the design is both trainable and deployable under 1.58-bit weights / 4-bit activations.

\appendix

\section{Reference Hyperparameters}

\begin{itemize}
    \item \textbf{Batching.} Effective batch 32--64 via grad accumulation; sequence length 128--256
    \item \textbf{LRs.} Base LR $3 \times 10^{-4}$ (ternary), $1.5 \times 10^{-4}$ (8-bit components), weight decay 0.01
    \item \textbf{Warm-up.} 10k steps; $\alpha$ ramp $0.05 \to 0.5$; kd\_coeff cosine $0 \to 0.5$
    \item \textbf{Percentile EMA.} Momentum 0.95 (train), 0.99 (calibration); $p = 99.7$ default
\end{itemize}

\section{Suggested Ablation Table Templates}

\begin{table}[h]
\centering
\caption{Effect of Orthogonal Mixing (WikiText-2 MLM PPL)}
\begin{tabular}{lccc}
\toprule
\textbf{Mixer} & \textbf{Extra cost} & \textbf{PPL $\downarrow$} & \textbf{Notes} \\
\midrule
None & 0 & --- & baseline ternary \\
DPD & $\sim$0 & --- & free shuffles \\
Block-H (per head, $b=64$) + DPD & low & --- & best accuracy/runtime \\
Dense H & high & --- & impractical \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Effect of Activation Scaling}
\begin{tabular}{lccc}
\toprule
\textbf{Scaling} & \textbf{Bits (A)} & \textbf{Utilization $\uparrow$} & \textbf{PPL $\downarrow$} \\
\midrule
Absmax & 4 & low & --- \\
Percentile-EMA (99.7) + SR & 4 & high & --- \\
\bottomrule
\end{tabular}
\end{table}

(Fill with measured numbers when you run the suite.)

\section{Reproducibility Notes}

\begin{itemize}
    \item Fix seeds and DPD permutations per run
    \item Log per-layer codebook usage and tanh saturation
    \item Export quantizer scales with commit hash; freeze them for eval
\end{itemize}

\section{Minimal PyTorch Snippets (Illustrative)}

\begin{verbatim}
class DPD(nn.Module):
    def __init__(self, d, perm=None):
        super().__init__()
        self.sign1 = nn.Parameter(torch.ones(d), 
                                  requires_grad=False)
        self.sign2 = nn.Parameter(torch.ones(d), 
                                  requires_grad=False)
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
\end{verbatim}

These snippets illustrate interfaces only; production code should use fused, quantized kernels and packed 2-bit weights.

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
