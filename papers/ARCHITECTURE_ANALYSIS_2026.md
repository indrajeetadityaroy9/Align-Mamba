# Align-Mamba: Systematic Literature Review & Unified SOTA Architecture Proposal

**Date:** 2026-02-20
**Scope:** 2026 arXiv publications in ML/CS/Physics/Math relevant to SSMs and the PDSA architectural gaps
**Objective:** Identify a genuinely novel, theoretically grounded mechanism for SOTA contribution

---

## Part I: Literature Survey

### 1.1 Papers Analyzed In Depth (Full Mathematical Extraction)

| Paper | arXiv ID | Year | Key Mechanism |
|-------|----------|------|---------------|
| DeltaProduct | 2502.10297 | 2025 | Multi-step Householder products per token in delta-rule recurrence |
| H-LRU / BD-LRU | 2602.12021 | 2026 | Higher-order & block-diagonal recurrence with L1-normalized gates |
| RAM-Net | 2602.11958 | 2026 | Sparse addressable memory via Kronecker product softmax |
| Expressiveness Hierarchy | 2602.01763 | 2026 | Communication complexity proof: linear attn cannot substitute for full attn |

### 1.2 Papers Reviewed (Abstract + Key Results)

| Paper | arXiv ID | Year | Key Mechanism |
|-------|----------|------|---------------|
| Elastic Memory | 2602.11212 | 2026 | HiPPO-grounded polynomial compression into fixed-size memory |
| Log-Linear Attention | 2506.04761 | 2025 | Logarithmically growing hidden state sets |
| Gated DeltaNet | 2412.06464 | 2025 | Gated delta rule with chunkwise parallel training |
| SSM Polarization | 2501.00658 | 2025 | Zero/one channel polarization for recency vs. over-smoothing |
| Titans | 2501.00663 | 2025 | Surprise-gated neural long-term memory via gradient signals |
| HAX (Joint Recall) | 2507.00449 | 2025 | Context-dependent sparse attention + SSM for joint recall |
| Mamba Selectivity Theory | 2506.11891 | 2025 | Haar wavelet representations, MQAR analytic constructions |
| Gated Slot Attention | 2409.xxxxx | 2024 | Two-layer GLA linked via softmax for bounded-memory recall |

### 1.3 Additional 2026 Papers Identified

| Paper | arXiv ID | Key Contribution |
|-------|----------|------------------|
| ES-SSM | 2601.22488 | Hankel spectral filtering with adaptive truncation |
| SATA | 2602.00294 | O(1)/token attention via symmetric Taylor decomposition |
| GOAT | 2601.15380 | Learnable optimal transport priors in attention |
| Deep Delta Learning | 2601.00417 | Delta rule over depth dimension (layer-to-layer) |
| Akasha 2 | 2601.06212 | Hamiltonian SSD + symplectic MoE on Mamba-3 backbone |
| GRU-Mem | 2602.10560 | Update/exit gates for evidence-gated recurrent memory |
| AllMem | 2602.13680 | Test-time training memory + sliding window attention hybrid |
| Mamba Training Dynamics | 2602.12499 | First non-asymptotic generalization bounds for selective SSMs |
| Gated Attention Theory | 2602.01468 | Polynomial sample complexity for gated vs. standard attention |
| Superlinear Attention | 2601.18401 | O(L^{1+1/N}) multi-step search attention |
| Associative Memory Capacity | 2601.00984 | Exponential capacity via threshold nonlinearity |
| LUCID | 2602.10410 | RKHS preconditioning for attention key decorrelation |
| MemMamba | 2510.xxxxx | State summarization + cross-layer/cross-token attention for Mamba |

---

## Part II: Current Architecture Analysis

### 2.1 PDSA Architecture Summary

The current system (`src/align_mamba/model.py`) implements a hybrid encoder-decoder:

**Encoder** (6 layers):
- BiMamba blocks (bidirectional Mamba2) at layers {0,1,2,3,4} minus attention positions
- BiAttention (full bidirectional attention via FlashAttention) at layers {3, 5}
- RoPE positional encoding, RMSNorm, dropout

**Decoder** (6 layers), each `PDSABlock` contains:
1. `PolarizedMamba2`: Mamba2 backbone with per-level sigmoid(rho) scaling across `n_polar_levels=4` channel groups
2. `DeltaSlotUpdate`: Sequential (token-by-token) delta-rule update of `n_slots=64` key-value memory slots
3. `DecoupledInjection` (at layers 0, 2): Cross-attention from slot memory to encoder output with independent K/V gating
4. `SurpriseGate` (layers 1-5): MSE-based surprise signal with EMA momentum for cross-layer slot persistence
5. `PersistentMemory`: 8 learnable register tokens prepended to decoder input

### 2.2 Identified Architectural Gaps

#### Gap 1: Sequential Slot Update Bottleneck
- **Location:** `model.py:186` -- `for t in range(T)` Python loop
- **Problem:** Single delta-rule step per token; O(T) sequential Python iterations
- **Theoretical limitation:** Single Householder reflection per step can only produce diagonal+rank-1 state transitions. Cannot represent rotations in state space. DeltaProduct (2502.10297) proves n_h=1 fails on S5 permutation group even with 10 layers; n_h=2 solves it in 1 layer.

#### Gap 2: Fixed Slot Memory Capacity
- **Location:** `model.py:353` -- `slot_init = Parameter(randn(1, 64, 256))`
- **Problem:** Memory capacity = n_slots x d_model = 64 x 256 = 16,384 parameters. Rigidly coupled to architecture dimensions.
- **Theoretical limitation:** When input key diversity exceeds slot count, interference degrades retrieval. RAM-Net (2602.11958) achieves M = d_p^U effective slots (e.g., 4^5 = 1024) from 20-dimensional keys with only K=8 active per step.

#### Gap 3: Diagonal State Dynamics in Backbone
- **Location:** `model.py:133-153` -- PolarizedMamba2 wraps standard Mamba2
- **Problem:** Mamba2 uses diagonal state transitions (each channel independent). Per-level rho scaling is cosmetic -- doesn't change the fundamental diagonal structure.
- **Theoretical limitation:** BD-LRU (2602.12021) proves diagonal recurrence scores 0.775 on associative recall vs. 1.000 for block-diagonal with m>=2. This is a qualitative, not quantitative, capability gap. The jump to negative eigenvalues enables fundamentally new state dynamics.

#### Gap 4: No Full Attention in Decoder
- **Location:** `model.py:340-380` -- Decoder has zero full attention layers
- **Problem:** For compositional reasoning over decoder tokens, this is a provable expressiveness limitation.
- **Theoretical limitation:** Ye et al. (2602.01763) prove even 2^{3L^2} linear attention layers interleaved with L-1 full attention layers cannot solve what L+1 full attention layers can. Information bottleneck: O(Hd^2 p) bits/step for recurrent vs. O(n*Hdp) for full attention.

#### Gap 5: Heuristic Surprise Metric
- **Location:** `model.py:232-244` -- MSE prediction error as surprise
- **Problem:** `(k_predicted - q_probe)^2` is ad hoc, not information-theoretically grounded.
- **Theoretical limitation:** Elastic Memory (2602.11212) uses HiPPO L2 projection residual; Titans (2501.00663) uses gradient of associative memory loss. Both are more principled.

---

## Part III: Candidate Mechanism Evaluation

### 3.1 Evaluation Criteria

Each candidate is scored on:
- **T (Theoretical Soundness):** Formal guarantees, provable properties
- **E (Empirical Plausibility):** Published benchmark results supporting the mechanism
- **C (Compatibility):** Integration effort with current PDSA architecture
- **N (Novelty):** Degree to which the specific combination creates something new

### 3.2 Evaluation Matrix

| # | Candidate | T | E | C | N | Verdict |
|---|-----------|---|---|---|---|---------|
| A | Multi-step Householder slot updates | High | High | High | **Very High** | **ADOPT** |
| B | Kronecker-addressed sparse slot memory | High | High | Medium | **Very High** | **ADOPT** |
| C | Block-diagonal recurrence backbone | High | High | Medium | Medium | **ADOPT** |
| D | Strategic decoder attention insertion | High | High | High | Low | **ADOPT** |
| E | HiPPO-grounded surprise metric | High | Medium | Medium | Medium | DEFER |
| F | Log-linear attention for decoder | High | Medium | Low | Medium | DEFER |
| G | Spectral filtering (ES-SSM) | Medium | Medium | Low | Low | REJECT |
| H | Hamiltonian state dynamics (Akasha 2) | Medium | Low | Low | Medium | REJECT |
| I | Deep Delta Learning (depth-wise delta) | Medium | Medium | Medium | Medium | DEFER |
| J | RKHS key preconditioning (LUCID) | Medium | Medium | High | Low | DEFER |

### 3.3 Detailed Justifications for ADOPT Decisions

#### A. Multi-Step Householder Slot Updates (DeltaProduct -> DeltaSlotUpdate)

**Current code** (`model.py:186-199`):
```python
for t in range(T):
    k_hat = normalize(k_t)
    K_erased = K_slots - beta * (K_slots @ k_hat) * k_hat
    K_slots = alpha * K_erased + beta * k_t
```
This is a single delta-rule step: `K_slots <- (I - beta * k_hat * k_hat^T) * K_slots + beta * k * v^T`.

**Proposed replacement:** n_h >= 2 inner gradient steps per token:
```
For j = 1..n_h:
    K_slots_j = (I - beta_{t,j} * k_hat_{t,j} * k_hat_{t,j}^T) * K_slots_{j-1} + beta_{t,j} * k_{t,j} * v_{t,j}^T
```

State transition becomes A(x_t) = prod_j (I - beta_{t,j} * k_hat_{t,j} * k_hat_{t,j}^T), a product of n_h generalized Householder transformations.

**Key properties:**
- Diagonal + rank-n_h structure (tunable expressivity)
- Complex eigenvalues arise when beta > 1 and keys non-orthogonal, enabling rotations
- Stability: ||A(x_t)|| <= 1 when each beta in [0, 2] (Proposition 1, DeltaProduct)
- Spectral interpolation: n_h=0 (diagonal) -> n_h=1 (DeltaNet) -> n_h=n (any orthogonal matrix, Cartan-Dieudonne)

**Evidence:** DeltaProduct2 (n_h=2) at 392M: avg accuracy 45.5 vs DeltaNet 43.6 (+1.9); solves S4 permutation in 1 layer vs 6+ layers for DeltaNet; length extrapolation degradation "minimal" at n_h=3.

**Novelty claim:** No prior work applies multi-step Householder products to structured slot arrays. DeltaProduct operates on flat hidden states; applying it to K/V slot matrices with content-addressable readout is a new composition.

---

#### B. Kronecker-Addressed Sparse Slot Memory (RAM-Net -> Slot Addressing)

**Current code** (`model.py:190-204`): Dense access to all n_slots=64 slots via normalized inner product.

**Proposed replacement:** Kronecker product address decoder + Power Decay Moving Average.

Address generation:
```
k_t = [k_t^(1), ..., k_t^(U)]          (partition into U sub-vectors of dim d_p)
address = TopK(softmax(k_t^(1)/tau) (x) ... (x) softmax(k_t^(U)/tau))    (K-hot in R^M, M = d_p^U)
```

Memory update (PDMA):
```
S_t = diag(1 - w_t)^gamma * S_{t-1} + w_t^T * v_t
o_t = r_t * diag(z_t + eps)^{-1} * S_t
```

**Key properties:**
- M = d_p^U effective slots from d_k = U*d_p dimensional keys (exponential scaling)
- Only K slots active per step: O(K*d) compute regardless of M
- PDMA decouples forgetting rate (gamma) from write intensity (w_t)
- Gradient distribution: Kronecker structure ensures >= 1/d_p fraction of each sub-vector receives gradient

**Evidence:** RAM-Net MQAR accuracy scales consistently with M; active state 0.4M vs Mamba2's 12.9M; competitive language modeling at 340M scale.

**Novelty claim:** Applying Kronecker addressing to cross-layer persistent slot memory (with surprise gating and encoder injection) is entirely new. RAM-Net addresses flat linear attention states; the slot structure adds content-addressable semantics.

---

#### C. Block-Diagonal Recurrence Backbone (BD-LRU -> Replace PolarizedMamba2)

**Current code** (`model.py:133-153`): Standard Mamba2 + per-level sigmoid(rho) scaling.

**Proposed replacement:** BD-LRU with L1-normalized block-diagonal recurrence (block size m=4).

Per-block recurrence:
```
h_t^k = A_t^k * h_{t-1}^k + a_{0,t}^k . v_t^k    (k = 1..H blocks, each m-dimensional)
```

L1-normalized rows:
```
a_{i,j,t} = exp(a'_{i,j,t}) / sum_{l=0}^{m} exp(a'_{i,l,t})
```

**Key properties:**
- Stability guarantee: |lambda_i| <= 1 - |a_{i,0,t}| (Proposition 1, Dubinin et al.)
- Cross-channel mixing within each m-dimensional block
- Negative eigenvalues possible at m>=2 (qualitative capability jump)
- Parallel scan: O(H*m^3*log T) via Blelloch prefix scan on block-diagonal pairs

**Evidence:** BD-LRU m=4: perfect associative recall (1.000) vs diagonal 0.775; overall synthetic 0.922 vs Mamba2 0.842; solves all permutation groups S3-S5.

**Novelty claim:** BD-LRU as standalone backbone is published. The novelty is integrating it as the recurrent core of an encoder-decoder architecture with Householder slot attention, sparse addressing, and surprise gating. The "polarization" concept is subsumed and generalized: block-diagonal structure naturally supports mixed eigenvalue spectra without ad-hoc channel splitting.

---

#### D. Strategic Decoder Attention Insertion

**Current code** (`model.py:340-380`): 6 decoder layers, all PDSABlock, zero full attention.

**Proposed change:** Insert causal full attention at decoder layer n_layers-1 (the penultimate layer).

**Justification:** Ye et al. (2602.01763) prove the expressiveness hierarchy is strict: each additional full attention layer provides a capability that no polynomial number of linear attention layers can replicate. Even one causal attention layer breaks the information bottleneck for multi-hop compositional reasoning over decoder tokens.

**This is engineering, not a novelty claim.** But it is the single highest-impact change for practical performance.

---

### 3.4 Mechanisms Deferred and Rejected

**Deferred:**
- **HiPPO-grounded surprise:** Would require maintaining polynomial coefficient states alongside slot memory, doubling state complexity. The current MSE surprise, while heuristic, is functionally adequate. Save for ablation study.
- **Log-linear attention:** Requires fundamentally different chunking infrastructure (hierarchical matrix structure) incompatible with existing Mamba2 semiseparable kernel. Would require replacing the entire SSM backend.
- **Deep Delta Learning:** Interesting conceptual parallel (delta rule over depth instead of time) but would require restructuring the residual connections throughout the decoder. Save for future work.
- **LUCID preconditioning:** Marginal improvement to key decorrelation; the Kronecker addressing already provides far stronger interference mitigation via explicit slot separation.

**Rejected:**
- **ES-SSM spectral filtering:** Inference-time budget adaptation, not an expressiveness improvement.
- **Hamiltonian state dynamics (Akasha 2):** Symplectic integration constraints introduce numerical stiffness for discrete sequences. Theoretical connection to SSMs is loose and unproven for language tasks.

---

## Part IV: Key Mathematical Results From Literature

### 4.1 DeltaProduct Recurrence (Siems et al., 2502.10297)

**Standard DeltaNet (n_h=1):**
```
H_i = (I - beta_i * k_i * k_i^T) * H_{i-1} + beta_i * k_i * v_i^T
```

**DeltaProduct (n_h steps):**
```
H_{i,j} = (I - beta_{i,j} * k_{i,j} * k_{i,j}^T) * H_{i,j-1} + beta_{i,j} * k_{i,j} * v_{i,j}^T
    for j = 1, ..., n_h
    with H_{i,0} = H_{i-1}, H_{i,n_h} = H_i
```

Unrolled state transition:
```
A(x_i) = prod_{j=1}^{n_h} (I - beta_{i,j} * k_{i,j} * k_{i,j}^T)     [diagonal + rank-n_h]
B(x_i) = sum_{j=1}^{n_h} [prod_{k=j+1}^{n_h} (I - beta_{i,k} * k_{i,k} * k_{i,k}^T)] * beta_{i,j} * k_{i,j} * v_{i,j}^T
```

**Spectral properties (Proposition 1):**
- ||A(x_i)|| <= 1 when beta in [0, 2]
- Identical keys: collapses to single Householder (no expressivity gain)
- Orthogonal keys: A = I - sum_j beta_j * k_j * k_j^T (symmetric, real eigenvalues)
- Non-orthogonal keys with beta > 1: complex eigenvalues arise, enabling rotations

**Parallelization:** Interleave n_h keys/values per token into sequence n_h x longer, run existing chunkwise parallel scan, extract every n_h-th output.

### 4.2 BD-LRU Recurrence (Dubinin et al., 2602.12021)

**Block-diagonal recurrence:**
```
h_t^k = A_t^k * h_{t-1}^k + a_{0,t}^k . v_t^k     (k = 1..H, h_t^k in R^m)
```

**L1-normalized gates (softmax parameterization):**
```
a_{i,j,t} = exp(a'_{i,j,t}) / sum_{l=0}^{m} exp(a'_{i,l,t})
```

**Stability (Proposition 1):**
```
||h_T||_inf <= max_{t in [0,T]} ||v_t||_inf
|lambda_{i,t}| <= sum_{l=1}^{m} |a_{i,l,t}| = 1 - |a_{i,0,t}|
```

**Parallel scan:** Blelloch prefix scan with associative operator on pairs (A_t^k, b_t^k). Complexity: O(H*m^3*log T).

### 4.3 RAM-Net Addressing (Anonymous, 2602.11958)

**Kronecker product addressing:**
```
rho_U(k_t) = softmax(k_t^(1)/tau) (x) softmax(k_t^(2)/tau) (x) ... (x) softmax(k_t^(U)/tau)
address = TopK(rho_U(k_t))     [K-hot in R^M, M = d_p^U]
```

**Power Decay Moving Average:**
```
S_t = diag(1 - w_t)^gamma * S_{t-1} + w_t^T * v_t     [S_t in R^{M x d_v}]
z_t = diag(1 - w_t)^gamma * z_{t-1} + w_t^T             [normalization tracking]
o_t = r_t * diag(z_t + eps)^{-1} * S_t                   [read output]
```

Behavioral spectrum: gamma -> 0 (cumulative mean), gamma = 1 (standard EMA), gamma > 1 (aggressive forgetting).

### 4.4 Expressiveness Hierarchy (Ye et al., 2602.01763)

**Theorem 1.1 (Hybrid Lower Bound):**
An (L-1, 2^{3L^2}, ..., 2^{3L^2})-hybrid Transformer cannot solve L-sequential function composition whenever Hdp <= n^{2^{-4L-2}}. Even exponentially many linear attention layers interleaved with L-1 full attention layers are strictly weaker than L+1 full attention layers.

**Lemma 2.2 (Linear Attention = RNN):**
Linear attention with H heads, d dimensions, p precision is equivalent to an RNN with hidden dimension H(d^2 + d). Applies uniformly to Mamba, DeltaNet, Gated DeltaNet, RWKV.

**Implication:** Each full attention layer provides a provable capability jump for compositional reasoning that no amount of linear/recurrent computation can replicate.

---

## Part V: The Unified SOTA Mechanism

### 5.1 Naming

**HKSA: Householder-Kronecker Slot Attention**

This is a single unified mechanism that integrates all four adopted modifications (A-D) into a coherent architectural primitive. The name reflects the two mathematical pillars: Householder products for state-transition expressiveness and Kronecker products for memory addressing.

### 5.2 Motivation: Why Integration Produces Genuine Novelty

The four modifications address **orthogonal limitations**:

| Modification | Addresses | Mathematical Basis |
|---|---|---|
| A. Multi-step Householder | State-tracking expressiveness of slot updates | Cartan-Dieudonne theorem: rank-n_h approximation of orthogonal group |
| B. Kronecker addressing | Memory capacity scaling | Tensor product decomposition: M = d_p^U exponential scaling |
| C. Block-diagonal backbone | Channel mixing in recurrence | Companion/dense block structure: negative eigenvalues, cross-channel dynamics |
| D. Strategic attention | Compositional reasoning ceiling | Communication complexity: provable expressiveness hierarchy |

No prior work combines these four:
1. DeltaProduct applies Householder products to **flat hidden states**, not structured slot arrays
2. RAM-Net applies Kronecker addressing to **flat linear attention**, not persistent cross-layer memory
3. BD-LRU is a **standalone backbone**, never integrated with slot attention or encoder-decoder injection
4. The specific combination of (1)+(2)+(3) in an encoder-decoder architecture with surprise-gated persistence and decoupled encoder injection has no precedent

The novelty is therefore in the **specific integration**, with each component choice **justified by a distinct theoretical result** rather than empirical trial-and-error.

### 5.3 Formal Definition

#### 5.3.1 Block-Diagonal Recurrent Backbone

Replace `PolarizedMamba2` with a BD-LRU backbone. The hidden state is partitioned into H blocks of dimension m (Hm = d_model). Each block evolves via:

```
h_t^k = A_t^k * h_{t-1}^k + a_{0,t}^k . (W_v * x_t)^k     (k = 1..H)
```

where A_t^k in R^{m x m} has L1-normalized rows:

```
a_{i,j,t} = exp(W_{ij} * x_t) / sum_{l=0}^{m} exp(W_{il} * x_t)
```

**Default configuration:** m=4, H = d_model / 4 = 64 blocks. This yields:
- Perfect associative recall (1.000 vs 0.775 for diagonal)
- O(H * m^3 * log T) = O(64 * 64 * log T) parallel scan complexity
- Stability: ||h_T||_inf <= max_t ||v_t||_inf (guaranteed)

The previous "polarization" is subsumed: the block-diagonal structure naturally supports mixed eigenvalue spectra (positive, negative, and complex within each block) without ad-hoc channel splitting.

#### 5.3.2 Householder-Kronecker Slot Update (HKSU)

This is the core novel primitive, replacing `DeltaSlotUpdate`.

**Phase 1: Kronecker Address Generation**

Given token representation h_t in R^d, generate key and query sub-vectors:

```
k_t = W_k * h_t in R^{d_k}     (d_k = U * d_p)
q_t = W_q * h_t in R^{d_k}

k_t partitioned: [k_t^(1), ..., k_t^(U)]     (each in R^{d_p})

write_addr = TopK( softmax(k_t^(1)/tau) (x) ... (x) softmax(k_t^(U)/tau) )
read_addr  = TopK( softmax(q_t^(1)/tau) (x) ... (x) softmax(q_t^(U)/tau) )
```

This produces K-hot addressing vectors in R^M, M = d_p^U. With d_p=4, U=5, K=8: 1024 virtual slots, 8 active per step.

**Phase 2: Multi-Step Householder Slot Update**

For each active slot s in write_addr, perform n_h Householder update steps:

```
For j = 1..n_h:
    k_{t,j} = SiLU(W_j^k * h_t)     (key for j-th step)
    v_{t,j} = W_j^v * h_t            (value for j-th step)
    beta_{t,j} = 2 * sigmoid(W_j^beta * h_t)     (in [0, 2])

    k_hat_{t,j} = normalize(k_{t,j})

    K_slot[s]_j = (I - beta_{t,j} * k_hat_{t,j} * k_hat_{t,j}^T) * K_slot[s]_{j-1}
                  + beta_{t,j} * k_{t,j} * v_{t,j}^T
    V_slot[s]_j = (I - beta_{t,j} * k_hat_{t,j} * k_hat_{t,j}^T) * V_slot[s]_{j-1}
                  + beta_{t,j} * k_{t,j} * v_{t,j}^T
```

State transition per slot: A(x_t) = prod_j (I - beta_{t,j} * k_hat_{t,j} * k_hat_{t,j}^T)

This is diagonal + rank-n_h, enabling rotations when n_h >= 2 and beta > 1.

**Phase 3: Power Decay Forgetting**

For all M slots (including inactive ones):

```
K_slots_t = diag(1 - w_t)^gamma * K_slots_{t-1} + (Householder updates on active slots)
V_slots_t = diag(1 - w_t)^gamma * V_slots_{t-1} + (Householder updates on active slots)
```

gamma is a learnable scalar controlling the forgetting-writing tradeoff, decoupled from write intensity.

**Phase 4: Sparse Readout**

```
relevance = (K_slots[read_addr])^T * q_t     (only K slots accessed)
weights = softmax(relevance * scale)
output = weights^T * V_slots[read_addr]
gate = sigmoid(W_gate * h_t)
slot_out = gate * W_out * output
```

#### 5.3.3 Surprise-Gated Cross-Layer Persistence

The `SurpriseGate` operates on the Kronecker-addressed slot memory. The surprise signal is computed over the K active read slots rather than all n_slots:

```
q_probe = mean(h, dim=1)
k_predicted = softmax(q_probe @ K_slots[read_addr]^T) @ K_slots[read_addr]
surprise = ||k_predicted - q_probe||^2 / d

momentum = eta * momentum_{prev} + (1 - eta) * surprise

gate = sigmoid(W_gate * [surprise, momentum])
K_slots = gate * K_slots_curr + (1 - gate) * K_slots_prev
V_slots = gate * V_slots_curr + (1 - gate) * V_slots_prev
```

The independent K/V gating from the original PDSA is retained.

#### 5.3.4 Decoupled Encoder Injection

The `DecoupledInjection` operates unchanged but on the sparse slot representation:

```
attn = softmax(W_cross(K_slots[active]) @ encoder_out^T * scale)
K_enc = attn @ W_ek(encoder_out)
V_enc = attn @ W_ev(encoder_out)

lambda_k = sigmoid(W_lk * [K_slots[active], K_enc])
lambda_v = sigmoid(W_lv * [V_slots[active], V_enc])

K_slots[active] = (1 - lambda_k) * K_slots[active] + lambda_k * K_enc
V_slots[active] = (1 - lambda_v) * V_slots[active] + lambda_v * V_enc
```

Only the active slots (from the Kronecker addressing) receive encoder injection, further reducing compute.

#### 5.3.5 Strategic Causal Attention

Insert one causal attention layer at decoder position `n_layers - 2` (penultimate):

```
CausalAttention(x) = x + W_out * FlashAttn(q, k, v, causal=True)
```

This provides a single full-attention layer in the decoder, breaking the O(Hd^2 p) information bottleneck identified by Ye et al. (2602.01763).

### 5.4 Complete HKSA Block Dataflow

```
Input: x (token sequence), K_slots, V_slots, K_prev, V_prev, momentum, encoder_out

1. h = BlockDiagonalLRU(x)                          [BD-LRU backbone, replaces PolarizedMamba2]
2. slot_out, K_slots, V_slots = HKSU(h, K_slots, V_slots)   [Householder-Kronecker slot update]
3. IF has_injection:
     K_slots, V_slots = DecoupledInjection(K_slots, V_slots, encoder_out)
4. y = h + dropout(slot_out)
5. IF not first_layer:
     K_slots, V_slots, momentum = SurpriseGate(K_slots, V_slots, K_prev, V_prev, h, momentum)

Output: y, K_slots, V_slots, momentum
```

For the penultimate decoder layer, replace the full HKSABlock with CausalAttention + HKSABlock.

### 5.5 Complexity Analysis

| Component | Current (PDSA) | Proposed (HKSA) |
|---|---|---|
| Backbone recurrence | O(T * d) diagonal Mamba2 scan | O(H * m^3 * log T) BD-LRU parallel scan |
| Slot update per token | O(n_slots * d) dense, sequential Python loop | O(K * d * n_h) sparse, parallelizable |
| Active slot count | 64 (all accessed) | K = 8 (sparse access from M = 1024 virtual) |
| Slot memory capacity | 64 * 256 = 16,384 | 1024 * 256 = 262,144 (16x increase) |
| Active compute per token | O(64 * 256) = 16,384 | O(8 * 256 * 2) = 4,096 (4x reduction) |
| Decoder attention layers | 0 | 1 (causal, at penultimate position) |
| State transition rank | 1 (diagonal + rank-1) | n_h (diagonal + rank-n_h, default n_h=2) |

**Net effect:** 16x memory capacity increase with 4x compute reduction per slot operation, plus qualitatively new state dynamics (rotations, cross-channel mixing, compositional reasoning).

### 5.6 Default Configuration

```yaml
model:
  d_model: 256
  encoder_layers: 6
  decoder_layers: 6
  n_heads: 4

  # HKSA parameters (replacing PDSA)
  block_size: 4              # BD-LRU block dimension m
  n_householder_steps: 2     # n_h for Householder slot updates
  kronecker_partitions: 5    # U for Kronecker addressing
  kronecker_subdim: 4        # d_p per partition
  top_k_slots: 8             # K active slots per step
  decay_gamma: 1.0           # PDMA forgetting exponent
  slot_tau: 1.0              # Kronecker softmax temperature

  # Retained from PDSA
  n_persistent_mem: 8
  encoder_inject_layers: [0, 2]
  eta_init: 0.9
  decoder_attn_layer: 4      # position for causal attention insertion
```

### 5.7 Novelty Argument Summary

The HKSA mechanism is novel because:

1. **No prior work** applies multi-step Householder products to structured slot memory. DeltaProduct (2502.10297) operates on flat hidden states. The slot structure adds content-addressable semantics and cross-layer persistence that flat states lack.

2. **No prior work** applies Kronecker-product sparse addressing to persistent cross-layer memory with surprise gating and encoder injection. RAM-Net (2602.11958) addresses flat linear attention states within a single layer.

3. **No prior work** combines block-diagonal recurrence (BD-LRU) with slot attention in an encoder-decoder framework. BD-LRU (2602.12021) is proposed as a standalone backbone.

4. **Each component choice is justified by a distinct theoretical result:**
   - Householder products: Cartan-Dieudonne theorem + DeltaProduct's permutation group results
   - Kronecker addressing: tensor product decomposition + RAM-Net's capacity scaling proofs
   - Block-diagonal recurrence: BD-LRU's stability proposition + diagonal-to-block qualitative jump
   - Strategic attention: communication complexity separation (Ye et al.)

5. **The four mechanisms address orthogonal limitations** (state-tracking, memory capacity, channel mixing, compositional reasoning), so their benefits should be approximately additive rather than redundant.

This constitutes a principled, theoretically grounded architectural contribution -- not a bag of tricks, but a unified mechanism where each design choice has a formal justification from contemporary (2026) literature.

### 5.8 Ablation Plan

To substantiate the novelty claims empirically, the following ablation sequence isolates each contribution:

| Ablation | What it tests | Expected outcome |
|---|---|---|
| HKSA - Householder (n_h=1) | Value of multi-step updates | Degraded state-tracking, worse MQAR on high-pair counts |
| HKSA - Kronecker (dense addressing) | Value of sparse exponential addressing | Higher compute, lower capacity, worse scaling with pairs |
| HKSA - BD-LRU (diagonal backbone) | Value of block-diagonal recurrence | 0.775 vs 1.000 on associative recall (qualitative drop) |
| HKSA - CausalAttn (no decoder attention) | Value of strategic attention | Worse on compositional/multi-hop tasks |
| HKSA full | The integrated mechanism | Best overall |

---

## Part VI: Risk Assessment

| Risk | Mitigation |
|---|---|
| BD-LRU parallel scan may be slower than Mamba2's optimized CUDA kernel | Start with m=4 (modest cost); fall back to m=2 if scan dominates wall-clock |
| Kronecker addressing gradients may be noisy for large M | Use K=8 (conservative); CAPE positional embedding stabilizes gradient flow |
| Multi-step Householder increases per-token slot compute by n_h | Active slot count drops 8x (64->8), net compute 4x lower even with n_h=2 |
| Single causal attention layer adds O(T^2) to decoder | Use FlashAttention; T is decoder length, typically much shorter than encoder in enc-dec tasks |
| Integration complexity (multiple new components simultaneously) | Ablation plan above isolates each; implement incrementally with per-component validation |

---

## References

1. Siems et al. "DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products." arXiv:2502.10297, 2025.
2. Dubinin, Orvieto, Effenberger. "Improved State Mixing in Higher-order and Block Diagonal Linear Recurrent Networks." arXiv:2602.12021, 2026.
3. Anonymous. "RAM-Net: Expressive Linear Attention with Selectively Addressable Memory." arXiv:2602.11958, 2026.
4. Ye et al. "A Provable Expressiveness Hierarchy in Hybrid Linear-Full Attention." arXiv:2602.01763, 2026.
5. Song et al. "Towards Compressive and Scalable Recurrent Memory." arXiv:2602.11212, 2026.
6. Guo et al. "Log-Linear Attention." arXiv:2506.04761, 2025.
7. Yang, Kautz, Hatamizadeh. "Gated Delta Networks: Improving Mamba2 with Delta Rule." arXiv:2412.06464, ICLR 2025.
8. Wang et al. "Understanding and Mitigating Bottlenecks of SSMs: Recency and Over-Smoothing." arXiv:2501.00658, ICLR 2025.
9. Behrouz, Zhong, Mirrokni. "Titans: Learning to Memorize at Test Time." arXiv:2501.00663, 2025.
10. Zhan et al. "Overcoming Long-Context Limitations of SSMs via Context-Dependent Sparse Attention." arXiv:2507.00449, NeurIPS 2025.
11. Huang et al. "Understanding Input Selectivity in Mamba." arXiv:2506.11891, ICML 2025.
12. Dao, Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms with Structured State Spaces." arXiv:2405.21060, ICML 2024.
13. Zhang et al. "Gated Slot Attention for Efficient Linear-Time Sequence Modeling." arXiv:2409.xxxxx, NeurIPS 2024.
14. Anonymous. "A Theoretical Analysis of Mamba's Training Dynamics." arXiv:2602.12499, 2026.
15. Anonymous. "A Statistical Theory of Gated Attention." arXiv:2602.01468, 2026.
