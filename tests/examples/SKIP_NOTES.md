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
