/**
 * @file script.hpp
 * @brief Minimal Python-subset scripting for the JIT.
 *
 * @see compile_script
 */

#pragma once

#include <memory>
#include <string>

#include <tenzor/jit/compiler.hpp>

namespace tenzor::jit {

/**
 * @brief Compile a Python-like script into a runnable CompiledModule.
 *
 * This is a scripting frontend — it parses a subset of Python sufficient
 * for short pure-computation kernels. The returned CompiledModule can be
 * invoked via `forward(Variable)` for single-argument scripts.
 *
 * **Supported grammar**
 * - A single top-level `def NAME(arg, ...):` function per script.
 * - A statement body of one or more statements:
 *   - `return EXPR` — ends the function.
 *   - `NAME = EXPR` — variable assignment (new or existing name).
 *   - `if COND: BODY` / `if COND: BODY else: BODY` — genuine multi-
 *     statement then/else blocks. `COND` uses one of the comparison
 *     operators below. Condition evaluation is NaN-safe (matches the
 *     project-wide eager/interpreter device-side-cast-order fix).
 *   - `for NAME in range(N): BODY` — a compile-time-constant-trip-count
 *     loop, unrolled at script-compile time (`N` must be a literal).
 * - `EXPR` may use:
 *   - Identifiers (function arguments or names assigned earlier in the
 *     body).
 *   - Floating-point and integer literals.
 *   - Binary arithmetic: `+`, `-`, `*`, `/` (left-associative, standard
 *     precedence: `*` / `/` above `+` / `-`).
 *   - Comparison: `<`, `>` (used in `if` conditions).
 *   - Method calls on a receiver expression, e.g. `x.sum()`, `x.mean()`
 *     (a fixed whitelist of tensor-reduction-style methods; see
 *     `src/jit/script.cpp`'s dispatch table for the exact set).
 *   - Parenthesised subexpressions.
 *
 * **Deliberately unsupported**
 * - Multiple functions per script.
 * - General Python control flow beyond the bounded `if`/`for` forms above
 *   (e.g. `while`, data-dependent-trip-count loops, `break`/`continue`).
 * - Arbitrary attribute access or method calls outside the whitelist.
 *
 * Any of these may be added as tests require them — the parser is a plain
 * recursive-descent parser in `src/jit/script.cpp`.
 *
 * @param source Null-terminated script source.
 * @return Compiled module whose `forward(Variable)` evaluates the script's
 *         return expression for its single argument.
 *
 * @throws std::runtime_error on lexer / parser / codegen errors, with a
 *         message pointing at the offending token.
 *
 * @code
 * auto compiled = tenzor::jit::compile_script(R"(
 *     def forward(x):
 *         return x * 2.0 + 1.0
 * )");
 * Variable y = compiled->forward(x);
 * @endcode
 */
auto compile_script(const char* source) -> std::shared_ptr<CompiledModule>;

/**
 * @brief Overload that specializes the compiled module for a given dummy input.
 *
 * The default `compile_script(source)` traces with a CPU+Float32 {1}-shape
 * dummy, which limits the resulting CompiledModule to that dtype/device. Pass
 * a matching `dummy` here (same dtype and device as your intended runtime
 * inputs) to specialise the traced graph for that combination. Shape does
 * not need to match exactly — the graph relies on broadcasting at runtime
 * — but dtype and device are captured.
 */
auto compile_script(const char* source, const Tensor& dummy)
    -> std::shared_ptr<CompiledModule>;

/**
 * @brief Compile and trace a multi-argument script (audit J6).
 *
 * Supports scripts with N > 1 input arguments. The number of dummies must
 * match the script's declared arg count. The returned CompiledModule's
 * `forward(std::vector<Variable>)` overload is the user-facing entry.
 *
 * Single-argument scripts can still use the `(source, dummy)` overload above.
 *
 * @param source The script source
 * @param dummies One example tensor per script argument (same dtype/device/
 *        shape as production inputs)
 */
auto compile_script(const char* source, const std::vector<Tensor>& dummies)
    -> std::shared_ptr<CompiledModule>;

} // namespace tenzor::jit
