# Attention Op Contract

This document codifies the cross-backend contract every fused/flash/scaled-dot-product
attention OpId must honor. Every backend kernel header references this file.

The contract was established after a six-agent cross-backend audit (CPU / CUDA / ROCm /
OneAPI / Vulkan) found the same architectural issues repeating across surfaces:
silently dropped causal flags, dropped dropout, returning the wrong number of tensors,
stride-from-shape indexing, and absent autograd integration. The single source of truth
below is the load-bearing fix — backends that drift will fail
`tests/integration/attention_contract_test.cpp`.

## Why

- The forward must always return saved state for backward — `logsumexp` (LSE) is the
  whole point of FlashAttention's memory-efficient backward; dropping it forces a full
  recompute. Dropout backward must be reproducible, which requires saving Philox
  `(seed, offset)` from the forward.
- `LSE` must be Float32 regardless of input dtype because it stores
  `max + log(sum(exp(s-max)))` whose dynamic range exceeds FP16/BF16.
- Causal masking must apply *before* softmax. Applying after, then renormalising, is
  numerically wrong (and historically what many of the silent-drop GPU paths fell back to).
- Stride correctness: kernels must accept Q/K/V with non-contiguous storage (typical
  output of `permute(0, 2, 1, 3)`). Either the host helper enforces `.contiguous()` at
  entry, or the kernel reads strides; never compute strides from shape.
- GQA / MQA: `H_kv` may be less than `H_q`. Kernels broadcast K/V along the head dim
  via `kv_h = q_h * H_kv / H_q` index math; the autograd plumbing must not sever
  K/V projection grads via raw `expand` + `reshape` rewraps.

## FlashAttention (`OpId::FlashAttention`, `OpId::FlashAttentionBackward`)

### Forward

| | |
|---|---|
| Inputs | `Q [..., S_q, D]`, `K [..., S_k, D]`, `V [..., S_k, D_v]`, optional `attn_mask` (additive, broadcastable to scores shape) |
| Attrs | `Scale: float` (multiplicative; default `1/sqrt(D)`), `Causal: bool`, `DropoutP: float`, `IsTraining: bool`, `Seed: int64` (Philox seed; if 0, backend draws from global RNG and reports the actual seed used in the output) |
| Outputs | `(output, logsumexp, philox_seed, philox_offset)` |
| Output 0 | `output [..., S_q, D_v]` — input dtype |
| Output 1 | `logsumexp [..., S_q]` — **always Float32**, holds `row_max + log(row_sum_exp)` per query |
| Output 2 | `philox_seed []` — Int64 scalar; the seed that was used (echoed input or generated). **Empty Tensor if `DropoutP == 0`** |
| Output 3 | `philox_offset []` — Int64 scalar; the starting Philox offset. **Empty Tensor if `DropoutP == 0`** |

### Backward

| | |
|---|---|
| Inputs | `dO, Q, K, V, O, L, philox_seed, philox_offset` |
| Attrs | `Scale, Causal, DropoutP` (no `IsTraining`; backward implies training) |
| Outputs | `(dQ, dK, dV)` — input dtype |

If `DropoutP > 0`, the backward must seed Philox at `(seed, offset)` and reproduce the
exact same dropout mask the forward used.

### Sentinels

- Causal mask sentinel: `-std::numeric_limits<float>::infinity()` (or
  `-numeric_limits<T>::infinity()` for the dtype). **Never `-1e9` or `-1e30`** — those
  saturate to `-65504` in FP16 and leak gradient mass through softmax.
- LSE sentinel for fully-masked rows: `-INFINITY`. Backward computes
  `P = exp(S - L)` which evaluates to 0 for `L = -inf`, correctly producing zero grads.

## FusedAttention (`OpId::FusedAttention`)

The cuDNN-SDPA-friendly variant. Same attrs as FlashAttention except dropout is
not supported.

| | |
|---|---|
| Inputs | `Q, K, V`, optional `attn_mask` |
| Attrs | `Scale, Causal` |
| Outputs | `(output, logsumexp)` — same dtype rules as FlashAttention |
| Constraint | Raise `std::invalid_argument` if `DropoutP > 0` is set; route those callers to `OpId::FlashAttention` |

There is no separate `FusedAttentionBackward` OpId. The autograd `FusedAttentionFunction`
recomputes from `(O, L)` using the FlashAttention backward kernel.

## NestedAttention (`OpId::NestedAttention`, `OpId::NestedAttentionBackward`)

Segmented (ragged) variant. `Q`, `K`, `V` are packed value tensors with separate
length/offset tensors.

| | |
|---|---|
| Inputs | `Q_values, K_values, V_values, Q_offsets, KV_offsets`, optional `attn_mask` |
| Attrs | `Scale, Causal` |
| Outputs | `(output, logsumexp)` |
| Causal-with-cache convention | When `Lq != Lkv` (cross-attention with KV cache + new query tokens), causal is `mask = ki > qi + (Lkv - Lq)`. Mask threshold is shifted by the cache length so existing cache always attends. Backends must agree on this convention. |

## FlexAttention (`OpId::FlexAttention`, `OpId::FlexAttentionBackward`)

Block-sparse attention with score modification.

| | |
|---|---|
| Inputs | `Q, K, V`, optional `block_mask` |
| Attrs | `Scale`, `ScoreModId: int` (an `OpId` value identifying the score modification op — must be serializable; **not** a `std::function`) |
| Outputs | `(output, logsumexp)` |
| Constraint | `score_mod` is encoded as an `OpId`. Backends that don't implement the requested `ScoreModId` raise `std::invalid_argument`. The CPU reference implements the standard library: `causal`, `alibi`, `sliding_window`, `relative_position_bias`, `rectangular`. |

## FusedLayerNorm (`OpId::FusedLayerNorm`, `OpId::FusedLayerNormBackward`)

| | |
|---|---|
| Forward inputs | `X, weight, bias` |
| Forward attrs | `Eps: float`, `NormalizedShape: int_list` (encoded as comma-separated string per `OpAttributes::get_int_list`) |
| Forward outputs | `(output, mean, rstd)` — **all three required**; mean and rstd are saved for backward |
| Mean/rstd dtype | **Always Float32** for FP16/BF16 inputs (rstd in FP16 overflows for `eps=1e-5, var≈0`) |

Backward inputs in canonical order: `[grad_output, input, weight, mean, rstd]`. Every
backend must accept this order; OneAPI's prior `[grad_output, input, mean, rstd, weight]`
is non-conforming.

## FusedRMSNorm (`OpId::FusedRMSNorm`, `OpId::FusedRMSNormBackward`)

| | |
|---|---|
| Forward inputs | `X, weight` |
| Forward attrs | `Eps`, `NormalizedShape` (read via `AttrKey::NormalizedShape`, **never** `shape().back()` — that silently truncates multi-dim normalisation) |
| Forward outputs | `(output, rrms)` — `rrms` is `1/sqrt(mean(x^2) + eps)` per row, **Float32** for FP16/BF16 inputs |

## FusedSoftmaxCrossEntropy (`OpId::FusedSoftmaxCrossEntropy`)

| | |
|---|---|
| Inputs | `logits [N, C], targets [N]` (Int32 or Int64) |
| Attrs | `Reduction: string` ("mean" / "sum" / "none"), `IgnoreIndex: int` (default -100, matching PyTorch), optional `Weight` |
| Outputs | `(loss, grad_logits)` when `compute_grad=true` (default); `(loss,)` otherwise |
| Constraint | The dispatcher front-end (`src/ops/fused_ops.cpp:fused_softmax_cross_entropy`) validates `targets ∈ [0, num_classes)` before dispatch. Backends MUST also bounds-check internally to prevent OOB device-memory access (OneAPI was the gap). On out-of-range, backend writes `NaN` for that row's loss and zero grad. |

## Optimizer fused steps (`OpId::FusedAdamStep` etc.)

For F64 paths: never downcast to float for `sqrt` / `exp` / `log`. Vulkan F64 shaders
must use `float64_t` `sqrt` (via `GL_EXT_shader_explicit_arithmetic_types_float64`)
or refine via Newton-Raphson in F64 precision:

```glsl
float64_t inv = float64_t(1.0) / sqrt(float(x));     // initial guess in float
inv = inv * (float64_t(1.5) - float64_t(0.5) * x * inv * inv);  // refine in F64
```

Mixed precision (F16/BF16 weights with FP32 master): the host helper passes the master
copy to the kernel, never the low-precision view. Backends that don't support F16/BF16
fused steps must pre-cast to F32 in the host adapter, not in the kernel.

## Autograd integration

Forward dispatch must NEVER be called raw from `nn::functional::*` or `nn::layers::*`.
Always route through an autograd `Function` subclass that:

1. Saves all forward inputs needed for backward via `save_for_backward(...)`.
2. Saves scalar context (`scale`, `causal`, `dropout_p`, `seed`, `offset`) as member fields.
3. Sets `next_functions` to inputs' `grad_fn`s, mapping 1:1 with backward output gradients.
4. Sets `input_variables` to the input Variables for `create_graph=true` second-order
   support.
5. For multi-output forwards (FlashAttention's 4-tuple, FusedAttention's 2-tuple),
   one shared `grad_fn` is set on every output Variable (the SVD pattern at
   `src/autograd/ops.cpp:1532-1554`).
6. Conditional attach: skip `grad_fn` only when both `!any_input.requires_grad()`
   AND `!is_grad_enabled()`. A Variable with `requires_grad=true` must always have a
   `grad_fn` if any input requires grad.

## Stride / contiguity

Every GPU host helper that takes Q/K/V must either:

- (a) Force `Q = Q.contiguous()`, `K = K.contiguous()`, `V = V.contiguous()` at entry, or
- (b) Read actual strides from each Tensor and pass them to the kernel.

Compositing `Q.reshape(...)` directly on a permuted view will throw at the reshape call
(or worse, succeed silently if the permute happens to leave a contiguous layout).
Always `.contiguous()` BEFORE `.reshape()`, never after.

The MultiheadAttention nn layer typically calls `.contiguous()` on its way down, but
direct `dispatch<OpId::FlashAttention>` callers (autograd, vmap, manual users) bypass it.

## Dispatch table audit

`tests/integration/attention_contract_test.cpp` enumerates every backend with
`tenzor::backend::is_op_supported(OpId, Device::Type)` for the OpIds in this contract.
Any backend missing any required OpId fails the test. This catches FlexAttention-style
"silently absent on this backend" gaps.

## Out-of-contract behavior (explicit)

- **Numerical equivalence** is to within `1e-3` for F32, `1e-2` for F16/BF16, `1e-6` for
  F64, against the CPU reference. Bit-exact equivalence is not required (FMA ordering
  differs across backends).
- **Performance**: this contract is correctness-only. Per-backend perf optimization is
  tracked separately.
