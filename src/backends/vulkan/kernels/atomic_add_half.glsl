// atomic_add_half.glsl — canonical packed-F16 / packed-BF16 CAS write helper
//
// GLSL does not allow SSBO buffers to be passed as parameters, so this file
// is REFERENCE DOCUMENTATION rather than an actual #include-able helper.
// Each .comp shader that packs F16 / BF16 values 2-per-uint32 must inline a
// variant of `writeF16`/`writeBF16` with its target SSBO baked in.
//
// =============================================================================
// X.7 / Y.11 / Y.16 saturation strategy
// =============================================================================
//
// PRIOR BUG: shaders used a bounded CAS loop (1024 iterations) followed by a
// non-atomic fallback store:
//
//     for (uint iter = 0; iter < 1024; iter++) {
//         expected = old_val;
//         new_val  = (expected & ~mask) | (bits << shift);
//         old_val  = atomicCompSwap(buf[word_idx], expected, new_val);
//         if (old_val == expected) return;
//     }
//     buf[word_idx] = new_val;            // <- BUG: clobbers neighbour half-word
//
// On saturation the fallback unconditionally wrote `new_val`, which contains
// stale bits for the OTHER half-word that another invocation had just CAS'd in.
// This silently corrupted unrelated diagonal/output entries in SVD/QR/Cholesky
// (X.7) and backward kernels for softmax/layer_norm/group_norm/rms_norm/
// adaptive_pool/avg_pool (Y.11/Y.16).
//
// FIX (chosen strategy A): raise the retry cap to 65536 and REMOVE the
// non-atomic fallback store entirely. 65536 retries is more than sufficient
// for any realistic contention on a 32-bit word that holds two packed halves
// (max useful concurrent writers per word = 2). The previous 1024 cap was a
// thinko — the only contention is on the OTHER half of the same uint, not
// on this half, so progress is guaranteed in O(workgroup_size) iterations.
// 65536 is a hard ceiling that exists only as a defence against driver bugs
// causing live-lock — under correct hardware the loop exits in <<100 iters.
//
// We do NOT add a status SSBO + host throw (strategy B): the loop bound is
// chosen high enough that exhaustion implies hardware/driver failure, not
// a workload-dependent saturation that needs to be reported per-call.
//
// =============================================================================
// Canonical write helper (copy into each shader, renaming the SSBO field)
// =============================================================================
//
// void writeF16(uint element_idx, float val) {
//     uint word_idx = element_idx / 2;
//     uint half_idx = element_idx & 1u;
//     uint bits  = packHalf2x16(vec2(val, 0.0)) & 0xFFFFu;
//     uint shift = half_idx * 16u;
//     uint mask  = 0xFFFFu << shift;
//     uint old_val = BUFFER_NAME[word_idx];
//     for (uint iter = 0u; iter < 65536u; iter++) {
//         uint expected = old_val;
//         uint new_val  = (expected & ~mask) | (bits << shift);
//         old_val = atomicCompSwap(BUFFER_NAME[word_idx], expected, new_val);
//         if (old_val == expected) return;
//     }
//     // Unreachable under correct hardware. Do NOT add a non-atomic fallback
//     // store here — it clobbers the neighbouring half-word.
// }
//
// BF16 variant: identical, but use packBF16 round-to-nearest-even instead of
// packHalf2x16 to convert `val` to 16-bit bits. Same CAS body / cap / no
// fallback.
