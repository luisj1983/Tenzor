# Showcase examples skipped from ExampleRegression

This file lists showcase autograd examples that were intentionally not wired
into `tests/examples/test_all_autograd_examples.cpp` and the reason for
each skip.

## 11_chat_ai

**Reason**: GRU seq2seq with Bahdanau attention (424-line file) plus an
interactive REPL after training. Already has a synthetic 6-pair fallback
when `--data` is omitted (see `load_pairs` returning a hardcoded list of
greetings if the file fails to open), so the data-loading concern from PR
2.1 isn't actually a blocker.

The genuine reason it's still Tier B (after re-evaluation in PR 2.1
extension): the autograd surface this example exercises (matmul, GRU
gates, Bahdanau attention via softmax+matmul, log_softmax cross-entropy)
is already covered by the wired examples — `07_rnn_sequence` covers GRU
backward, `16_self_attention` covers attention backward, every example
covers cross-entropy. Adding chat_ai would extract a 50+ line runner that
duplicates the helpers (CharVocab, gru_step, bahdanau_context, sampling)
without surfacing a new bug class.

If a future Tenzor change introduces a feature unique to chat_ai
(e.g., changes to char-level tokenization or the specific Bahdanau
formulation), wire it then. The synthetic-fallback path in `load_pairs`
already makes the runner test-friendly — it's purely the runner-extraction
labour that's deferred.

The standalone showcase exe still builds and runs as before.

**audit-4 W.28 (2026-05-24)**: re-evaluated; remains intentional. The
autograd surface (GRU gates, Bahdanau attention, log_softmax cross-entropy)
is still fully covered by `07_rnn_sequence` and `16_self_attention`. No
new chat_ai-specific tensor logic has landed since audit-3, so the
deferred runner is still the right call.

## bert_zero_stage1 (audit-9 KK.27, 2026-05-26)

**Reason**: the example is intentionally disabled in
`examples/CMakeLists.txt` (lines 97-98) pending the implementation of
`tenzor/models/bert.hpp`, `distributed::get_default_process_group()`, and
`nn::optim::AdamW`. Wiring an autograd_runner regression test would
require those same dependencies, so the runner extraction is deferred
until the upstream prerequisites land. Once the executable target is
re-enabled, follow the pattern used in `gradient_checkpointing` /
`vit_image_classification` to extract a runner and wire it into
`tests/examples/test_all_autograd_examples.cpp`.

## Non-showcase examples covered (audit-9 KK.27, 2026-05-26)

- `examples/cpp/training/vit_image_classification.cpp` -> runner extracted
  into `vit_image_classification_runner.{cpp,hpp}`; wired via
  `NON_SHOWCASE_AUTOGRAD_RUNNER_TARGETS` in `tests/examples/CMakeLists.txt`
  and exercised by `ExampleRegression.VitImageClassificationTrains`.
- `examples/cpp/training/gradient_checkpointing.cpp` -> runner extracted
  into `gradient_checkpointing_runner.{cpp,hpp}`; wired via the same list
  and exercised by `ExampleRegression.GradientCheckpointingTrains`.
