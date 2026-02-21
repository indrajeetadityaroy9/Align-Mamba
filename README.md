# SlotMamba: Mitigating State-Capacity Bottlenecks via Householder-Kronecker Slot Attention

Selective state space models (SSMs) provide linear-time sequence processing, but their fixed-size latent state imposes a hard memory bottleneck: when the number of key-value associations exceeds `d_state`, retrieval accuracy degrades sharply due to state saturation. SlotMamba is a hybrid encoder-decoder architecture that addresses this state-capacity bottleneck through **Householder-Kronecker Slot Attention (HKSA)**, a recurrent slot-memory mechanism with five interlocking components:

1. **Kronecker-factored sparse addressing** - A Kronecker product of U learned softmax vectors generates M = d_p^U virtual memory slots, from which K active slots are selected via top-k. This factorization enables exponentially large address spaces with linear parameter cost.
2. **Multi-step Householder slot updates** - Active slots are updated via composition of Householder reflections with learned step sizes, providing expressive rank-modifying writes that generalize the delta rule.
3. **Power-decay memory allocation (PDMA)** - A learned power-decay forgetting mechanism tied to write addresses provides smooth, content-dependent memory deallocation with proper normalization tracking.
4. **Surprise-gated slot persistence** - An EMA-based surprise signal with learned K/V weighting gates slot updates across layers, allowing the model to selectively preserve or overwrite slot contents based on prediction error.
5. **Decoupled encoder injection** - Active write slots cross-attend to encoder output with independent K/V gating, enabling selective integration of encoder information into slot memory.

The encoder uses bidirectional Mamba2 (BiMamba) with strategic full-attention layers; the decoder threads slot state (K/V memory, PDMA normalization, surprise momentum) across all layers, with a block-diagonal linear recurrent unit (BD-LRU) backbone and a single causal attention layer at the penultimate position. The system preserves O(T) recurrent efficiency while scaling associative recall capacity beyond the SSM state bottleneck.

## Model

- Encoder: BiMamba with attention at `encoder_attn_layers`.
- Decoder: HKSA blocks with BD-LRU backbone.
- Slot memory: Kronecker top-k addressing, Householder writes, PDMA decay and normalization.
- Sparse control: encoder injection on selected layers, surprise gating after layer 0, single causal attention layer at `decoder_attn_layer`.

## References

1. *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products*. https://arxiv.org/abs/2502.10297
2. *RAM-Net: Expressive Linear Attention with Selectively Addressable Memory*. https://arxiv.org/abs/2602.11958
3. *Improved state mixing in higher-order and block diagonal linear recurrent networks*. https://arxiv.org/abs/2602.12021
4. *A Provable Expressiveness Hierarchy in Hybrid Linear-Full Attention*. https://arxiv.org/abs/2602.01763
5. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. https://arxiv.org/abs/2312.00752
6. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2). https://arxiv.org/abs/2405.21060
7. *Gated Slot Attention for Efficient Linear-Time Sequence Modeling*. https://arxiv.org/abs/2409.07146
8. *Gated Delta Networks: Improving Mamba2 with Delta Rule*. https://arxiv.org/abs/2412.06464
9. *Titans: Learning to Memorize at Test Time*. https://arxiv.org/abs/2501.00663
10. *Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing*. https://arxiv.org/abs/2501.00658
11. *Log-Linear Attention*. https://arxiv.org/abs/2506.04761
12. *Understanding Input Selectivity in Mamba: Impact on Approximation Power, Memorization, and Associative Recall Capacity*. https://arxiv.org/abs/2506.11891
13. *Overcoming Long-Context Limitations of State-Space Models via Context-Dependent Sparse Attention*. https://arxiv.org/abs/2507.00449
14. *Towards Compressive and Scalable Recurrent Memory*. https://arxiv.org/abs/2602.11212
