# Showcase examples skipped from ExampleRegression

This file lists showcase autograd examples that were intentionally not wired
into `tests/examples/test_all_autograd_examples.cpp` and the reason for
each skip.

## (no showcase examples currently deferred)

**audit-9 LL.19 (2026-05-26)**: `11_chat_ai` was previously listed here as
deferred. It was wired into `ExampleRegression` by audit-8 II.15 — its
runner now lives at `examples/cpp/showcase/11_chat_ai/autograd_runner.{cpp,hpp}`
and is invoked from `tests/examples/test_all_autograd_examples.cpp`. The
historical rationale (audit-3 / audit-4 W.28 re-evaluation) is preserved in
git history; the deferral itself is resolved.

## Non-showcase examples deferred (audit-9 KK.27, 2026-05-26)

The following non-showcase examples are intentionally NOT wired into
`ExampleRegression` because their build targets are disabled in
`examples/CMakeLists.txt` pending upstream dependencies. Wiring an
autograd_runner would require those same dependencies, so the runner
extraction is deferred until the executable target is re-enabled.

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

## Non-showcase examples covered (audit-10 NN.24, 2026-05-26)

- `examples/cpp/training/transformer_seq2seq.cpp` -> runner extracted
  into `transformer_seq2seq_runner.{cpp,hpp}`; wired via
  `NON_SHOWCASE_AUTOGRAD_RUNNER_TARGETS` and exercised by
  `ExampleRegression.TransformerSeq2SeqTrains`.
- `examples/cpp/training/yolo_object_detection.cpp` -> runner extracted
  into `yolo_object_detection_runner.{cpp,hpp}`; wired via the same list
  and exercised by `ExampleRegression.YoloObjectDetectionTrains`. The
  test asserts `initial != final` (per the plan's fallback) rather than
  monotonic decrease, because the simplified MSE-against-zeros surrogate
  loss is not guaranteed to decrease on a few iterations of a randomly
  initialised conv backbone.
- `examples/cpp/training/gpt_text_generation.cpp` -> runner extracted
  into `gpt_text_generation_runner.{cpp,hpp}`; wired via the same list
  and exercised by `ExampleRegression.GptTextGenerationTrains`. NOTE:
  the NN.24 plan referenced `examples/cpp/nlp/gpt_text_generation.cpp`,
  but that file is a pure-inference example (greedy/top-k/top-p/beam
  decoding with no training loop). The training-side counterpart lives
  at `examples/cpp/training/gpt_text_generation.cpp` (executable target
  `gpt_text_generation_training`) and is what this runner extracts from
  — it exercises the same GPTBlock + MultiheadAttention + LayerNorm +
  GELU + CrossEntropyLoss + Adam pipeline a real training regression
  needs to cover. The nlp/ inference example is not training, so no
  loss-decrease assertion applies; its TextGenerator strategies are
  covered by other unit tests.
