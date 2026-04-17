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
 * This is a minimum-viable implementation of a scripting frontend — it parses
 * a small subset of Python sufficient for short pure-computation kernels. The
 * returned CompiledModule can be invoked via `forward(Variable)` for
 * single-argument scripts.
 *
 * **Supported grammar (MVP)**
 * - A single top-level `def NAME(arg, ...):` function.
 * - Inside the function body, a single `return EXPR` statement.
 * - `EXPR` may use:
 *   - Identifiers (refer to function arguments only).
 *   - Floating-point and integer literals.
 *   - Binary arithmetic: `+`, `-`, `*`, `/` (left-associative, standard
 *     precedence: `*` / `/` above `+` / `-`).
 *   - Parenthesised subexpressions.
 *
 * **Deliberately unsupported in this MVP**
 * - Control flow (`if`, `else`, loops).
 * - Variable assignment (beyond function parameters).
 * - Attribute / method access (e.g. `x.mean()`).
 * - Multiple statements per function.
 * - Multiple functions per script.
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

} // namespace tenzor::jit
