/**
 * @file script.cpp
 * @brief Python-subset script compiler (see include/tenzor/jit/script.hpp).
 *
 * Grammar:
 *
 *   function   := 'def' IDENT '(' [IDENT (',' IDENT)*] ')' ':' NEWLINE
 *                 INDENT stmt+ DEDENT
 *   stmt       := assign | return | if_stmt | for_stmt
 *   assign     := IDENT '=' expr NEWLINE
 *   return     := 'return' expr NEWLINE
 *   if_stmt    := 'if' expr ':' NEWLINE INDENT stmt+ DEDENT
 *                 ['else' ':' NEWLINE INDENT stmt+ DEDENT]
 *   for_stmt   := 'for' IDENT 'in' 'range' '(' NUMBER ')' ':' NEWLINE
 *                 INDENT stmt+ DEDENT
 *   expr       := term (cmp_op term)?      # at most one comparison
 *   cmp_op     := '<' | '>'
 *   term       := mul_term (('+' | '-') mul_term)*
 *   mul_term   := factor (('*' | '/') factor)*
 *   factor     := unary method_suffix*
 *   unary      := NUMBER | IDENT | '(' expr ')' | '-' factor
 *   method_suffix := '.' IDENT '(' ')'
 *
 * Control flow semantics (trace-based JIT):
 *  - `if`/`else`: the condition is data-dependent and not representable in the
 *    IR, so reading it to pick a branch is a graph break. It is read through
 *    Tensor::item(), which fires the tracer's graph-break hook — under
 *    TENZOR_JIT_STRICT this throws; otherwise it warns and the taken branch's
 *    ops are baked into the compiled graph (standard trace-based semantic).
 *  - `for i in range(N)` (constant N): statically unrolled, so each iteration
 *    produces a distinct chain in the traced graph. This is fully captured (no
 *    data-dependent condition is read), so it is not a graph break.
 */

#include <tenzor/jit/script.hpp>

#include <cctype>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tenzor/jit/tracer.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/tenzor.hpp>

namespace tenzor::jit {
namespace {

// ============================================================================
// Lexer
// ============================================================================

enum class Tok {
    Def, Ident, Number, Return, If, Else, For, In, Range,
    LParen, RParen, Comma, Colon,
    Plus, Minus, Star, Slash,
    Less, Greater,
    Equals, Dot, Newline,
    Indent, Dedent,
    End
};

struct Token {
    Tok kind;
    std::string text;
    double number = 0.0;
    int line = 1;
    int col = 1;
};

[[noreturn]] void fail(const Token& t, std::string_view msg) {
    std::ostringstream os;
    os << "compile_script: " << msg << " at line " << t.line << ", col " << t.col
       << " (near '" << t.text << "')";
    throw std::runtime_error(os.str());
}

class Lexer {
public:
    explicit Lexer(std::string_view src) : src_(src) {}

    std::vector<Token> tokenize() {
        std::vector<Token> out;
        // Initialize indent stack with the first non-blank line's leading
        // whitespace so that raw-string literals (R"( ... )") starting with a
        // newline + indentation don't look like an anomalous INDENT.
        indent_stack_.push_back(detect_first_indent());
        bool at_line_start = true;

        while (pos_ < src_.size()) {
            if (at_line_start) {
                handle_indentation(out);
                at_line_start = false;
                if (pos_ >= src_.size()) break;
            }

            char c = src_[pos_];
            if (c == '#') {
                while (pos_ < src_.size() && src_[pos_] != '\n') advance();
                continue;
            }
            if (c == '\n') {
                out.push_back(make_tok(Tok::Newline, "\\n"));
                advance();
                at_line_start = true;
                continue;
            }
            if (c == ' ' || c == '\t' || c == '\r') { advance(); continue; }

            if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                out.push_back(read_identifier());
            } else if (std::isdigit(static_cast<unsigned char>(c)) ||
                       (c == '.' && pos_ + 1 < src_.size() &&
                        std::isdigit(static_cast<unsigned char>(src_[pos_+1])))) {
                out.push_back(read_number());
            } else {
                out.push_back(read_symbol());
            }
        }

        // Emit trailing DEDENTs to close any open blocks.
        while (indent_stack_.size() > 1) {
            out.push_back(make_tok(Tok::Dedent, "<dedent>"));
            indent_stack_.pop_back();
        }
        out.push_back(make_tok(Tok::End, ""));
        return out;
    }

private:
    std::string_view src_;
    size_t pos_ = 0;
    int line_ = 1;
    int col_ = 1;
    std::vector<int> indent_stack_;

    int detect_first_indent() const {
        size_t i = 0;
        while (i < src_.size()) {
            size_t line_start = i;
            int spaces = 0;
            while (i < src_.size() && (src_[i] == ' ' || src_[i] == '\t')) {
                spaces += (src_[i] == '\t') ? 8 : 1;
                ++i;
            }
            if (i >= src_.size()) return 0;
            if (src_[i] == '\n') { ++i; continue; }
            if (src_[i] == '#') { while (i < src_.size() && src_[i] != '\n') ++i; continue; }
            (void)line_start;
            return spaces;
        }
        return 0;
    }

    void advance() {
        if (pos_ < src_.size() && src_[pos_] == '\n') { ++line_; col_ = 1; }
        else { ++col_; }
        ++pos_;
    }

    Token make_tok(Tok k, std::string text) {
        Token t; t.kind = k; t.text = std::move(text); t.line = line_; t.col = col_;
        return t;
    }

    void handle_indentation(std::vector<Token>& out) {
        // Skip blank lines and comment-only lines (they don't affect indent).
        while (pos_ < src_.size()) {
            size_t line_start = pos_;
            int spaces = 0;
            while (pos_ < src_.size() && (src_[pos_] == ' ' || src_[pos_] == '\t')) {
                spaces += (src_[pos_] == '\t') ? 8 : 1;
                advance();
            }
            if (pos_ >= src_.size()) return;
            if (src_[pos_] == '\n') {
                out.push_back(make_tok(Tok::Newline, "\\n"));
                advance();
                continue;
            }
            if (src_[pos_] == '#') {
                while (pos_ < src_.size() && src_[pos_] != '\n') advance();
                continue;
            }
            // Real content line — emit INDENT / DEDENT tokens as needed.
            int current = indent_stack_.back();
            if (spaces > current) {
                indent_stack_.push_back(spaces);
                out.push_back(make_tok(Tok::Indent, "<indent>"));
            } else {
                while (spaces < indent_stack_.back()) {
                    indent_stack_.pop_back();
                    out.push_back(make_tok(Tok::Dedent, "<dedent>"));
                }
                if (spaces != indent_stack_.back()) {
                    Token t = make_tok(Tok::End, std::string(1, src_[line_start]));
                    fail(t, "inconsistent indentation");
                }
            }
            return;
        }
    }

    Token read_identifier() {
        int line = line_, col = col_;
        size_t start = pos_;
        while (pos_ < src_.size() &&
               (std::isalnum(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_')) {
            advance();
        }
        std::string name(src_.substr(start, pos_ - start));
        Tok kind = Tok::Ident;
        if (name == "def")         kind = Tok::Def;
        else if (name == "return") kind = Tok::Return;
        else if (name == "if")     kind = Tok::If;
        else if (name == "else")   kind = Tok::Else;
        else if (name == "for")    kind = Tok::For;
        else if (name == "in")     kind = Tok::In;
        else if (name == "range")  kind = Tok::Range;
        Token t; t.kind = kind; t.text = std::move(name); t.line = line; t.col = col;
        return t;
    }

    Token read_number() {
        int line = line_, col = col_;
        size_t start = pos_;
        bool seen_dot = false;
        while (pos_ < src_.size()) {
            char c = src_[pos_];
            if (std::isdigit(static_cast<unsigned char>(c))) { advance(); }
            else if (c == '.' && !seen_dot) { seen_dot = true; advance(); }
            else break;
        }
        std::string text(src_.substr(start, pos_ - start));
        Token t;
        t.kind = Tok::Number;
        t.text = text;
        t.number = std::strtod(text.c_str(), nullptr);
        t.line = line; t.col = col;
        return t;
    }

    Token read_symbol() {
        int line = line_, col = col_;
        char c = src_[pos_];
        std::string text(1, c);
        Tok kind;
        switch (c) {
            case '(': kind = Tok::LParen; break;
            case ')': kind = Tok::RParen; break;
            case ',': kind = Tok::Comma;  break;
            case ':': kind = Tok::Colon;  break;
            case '+': kind = Tok::Plus;   break;
            case '-': kind = Tok::Minus;  break;
            case '*': kind = Tok::Star;   break;
            case '/': kind = Tok::Slash;  break;
            case '=': kind = Tok::Equals; break;
            case '.': kind = Tok::Dot;    break;
            case '<': kind = Tok::Less;   break;
            case '>': kind = Tok::Greater;break;
            default: {
                Token t = make_tok(Tok::End, text);
                fail(t, std::string("unexpected character '") + c + "'");
            }
        }
        advance();
        Token t; t.kind = kind; t.text = text; t.line = line; t.col = col;
        return t;
    }
};

// ============================================================================
// AST
// ============================================================================

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;
struct Stmt;
using StmtPtr = std::shared_ptr<Stmt>;

struct NumberExpr     { double value; };
struct IdentExpr      { std::string name; };
struct BinOpExpr      { char op; ExprPtr lhs; ExprPtr rhs; };
struct CmpOpExpr      { char op; ExprPtr lhs; ExprPtr rhs; };  // '<' or '>'
struct MethodCallExpr { ExprPtr receiver; std::string method; std::vector<ExprPtr> args; };

struct Expr {
    std::variant<NumberExpr, IdentExpr, BinOpExpr, CmpOpExpr, MethodCallExpr> node;
};

struct AssignStmt { std::string target; ExprPtr value; };
struct ReturnStmt { ExprPtr value; };
struct IfStmt     { ExprPtr cond; std::vector<Stmt> then_body; std::vector<Stmt> else_body; };
struct ForStmt    { std::string var; int64_t count; std::vector<Stmt> body; };

struct Stmt {
    std::variant<AssignStmt, ReturnStmt, IfStmt, ForStmt> node;
};

struct FuncDef {
    std::string name;
    std::vector<std::string> args;
    std::vector<Stmt> body;
};

// ============================================================================
// Parser
// ============================================================================

class Parser {
public:
    explicit Parser(std::vector<Token> tokens) : toks_(std::move(tokens)) {}

    FuncDef parse_function() {
        FuncDef f;
        skip_newlines();
        expect(Tok::Def, "expected 'def'");
        const Token& name_tok = expect(Tok::Ident, "expected function name after 'def'");
        f.name = name_tok.text;
        expect(Tok::LParen, "expected '(' after function name");
        if (peek().kind != Tok::RParen) {
            f.args.push_back(expect(Tok::Ident, "expected argument name").text);
            while (peek().kind == Tok::Comma) {
                consume();
                f.args.push_back(expect(Tok::Ident, "expected argument name after ','").text);
            }
        }
        expect(Tok::RParen, "expected ')' closing argument list");
        expect(Tok::Colon, "expected ':' after function signature");
        // Accept both indented block `def f(x):\n    return x` and inline
        // `def f(x): return x` — the inline form is handy for tests and tiny
        // scripts.
        if (peek().kind == Tok::Return ||
            (peek().kind == Tok::Ident && peek(1).kind == Tok::Equals)) {
            f.body.push_back(parse_stmt());
        } else {
            skip_newlines();
            expect(Tok::Indent, "expected indented block for function body");
            f.body = parse_block();
        }
        ensure_has_return(f.body);
        skip_newlines();
        if (peek().kind != Tok::End) {
            fail(peek(), "unexpected token after function body");
        }
        return f;
    }

private:
    std::vector<Token> toks_;
    size_t pos_ = 0;

    // Hardening: bound recursion depth of the recursive-descent parser so that
    // a crafted source with deeply nested parentheses / method chains / nested
    // if-for blocks (e.g. "((((...))))" or "f(f(f(...)))" thousands deep) throws
    // a clean error instead of overflowing the native C++ stack (DoS / crash on
    // untrusted input). 256 comfortably exceeds any legitimate hand-written
    // script while staying far below the typical 1–8 MB stack budget.
    static constexpr int kMaxRecursionDepth = 256;
    int depth_ = 0;

    // RAII depth guard: increments on construction, throws if the limit is
    // exceeded, decrements on scope exit. Placed at the top of every mutually
    // recursive parse entry point.
    struct DepthGuard {
        Parser& p;
        explicit DepthGuard(Parser& parser) : p(parser) {
            if (++p.depth_ > kMaxRecursionDepth) {
                fail(p.peek(),
                     "expression/statement nesting too deep (max " +
                         std::to_string(kMaxRecursionDepth) + ")");
            }
        }
        ~DepthGuard() { --p.depth_; }
    };

    // Bounds-safe lookahead. The lexer always appends a single End token, but a
    // one-token lookahead (peek(1)) at the final real token, or any off>0 past
    // End, would index past the vector (OOB read on untrusted input). Clamp to
    // the last token (always End) so the parser sees a well-formed terminator
    // rather than reading out of bounds.
    const Token& peek(size_t off = 0) const {
        size_t i = pos_ + off;
        if (i >= toks_.size()) return toks_.back();
        return toks_[i];
    }
    const Token& consume() {
        if (pos_ >= toks_.size()) return toks_.back();  // End sentinel
        return toks_[pos_++];
    }

    const Token& expect(Tok kind, std::string_view msg) {
        if (peek().kind != kind) fail(peek(), msg);
        return consume();
    }

    void skip_newlines() {
        while (peek().kind == Tok::Newline) consume();
    }

    void ensure_has_return(const std::vector<Stmt>& body) {
        if (body.empty()) fail(peek(), "function body must end with 'return EXPR'");
        const auto& last = body.back();
        if (std::holds_alternative<ReturnStmt>(last.node)) return;
        // Inside if/else both arms must return (simplified: top-level must end with return).
        fail(peek(), "function body must end with 'return EXPR'");
    }

    std::vector<Stmt> parse_block() {
        DepthGuard guard(*this);  // bound nested if/for block depth
        std::vector<Stmt> stmts;
        while (true) {
            skip_newlines();
            if (peek().kind == Tok::Dedent) { consume(); break; }
            if (peek().kind == Tok::End) break;
            stmts.push_back(parse_stmt());
        }
        return stmts;
    }

    Stmt parse_stmt() {
        DepthGuard guard(*this);  // bound recursion via nested if/for bodies
        if (peek().kind == Tok::Return) {
            consume();
            ExprPtr v = parse_expression();
            // Consume trailing newline(s).
            if (peek().kind == Tok::Newline) consume();
            return Stmt{ReturnStmt{std::move(v)}};
        }
        if (peek().kind == Tok::If) {
            consume();
            ExprPtr c = parse_expression();
            expect(Tok::Colon, "expected ':' after if condition");
            skip_newlines();
            expect(Tok::Indent, "expected indented block after 'if'");
            auto then_body = parse_block();
            std::vector<Stmt> else_body;
            skip_newlines();
            if (peek().kind == Tok::Else) {
                consume();
                expect(Tok::Colon, "expected ':' after 'else'");
                skip_newlines();
                expect(Tok::Indent, "expected indented block after 'else'");
                else_body = parse_block();
            }
            return Stmt{IfStmt{std::move(c), std::move(then_body), std::move(else_body)}};
        }
        if (peek().kind == Tok::For) {
            consume();
            std::string var = expect(Tok::Ident, "expected loop variable after 'for'").text;
            expect(Tok::In, "expected 'in' after loop variable");
            expect(Tok::Range, "only 'range(N)' is supported for loop iterators");
            expect(Tok::LParen, "expected '(' after 'range'");
            const Token& n_tok = expect(Tok::Number, "expected integer literal inside 'range(...)'");
            expect(Tok::RParen, "expected ')' closing 'range(...)'");
            expect(Tok::Colon, "expected ':' after for header");
            skip_newlines();
            expect(Tok::Indent, "expected indented block after 'for'");
            auto body = parse_block();
            int64_t count = static_cast<int64_t>(n_tok.number);
            if (count < 0) fail(n_tok, "range(N) requires N >= 0");
            // Hardening: `for i in range(N)` is statically unrolled into N copies
            // of the loop body in the traced graph (exec_one below). A crafted
            // large N (e.g. range(2000000000)) would unroll into billions of
            // graph nodes — an OOM / unbounded-work DoS. Cap N at parse time so
            // untrusted scripts can't drive an arbitrarily large unroll.
            constexpr int64_t kMaxUnroll = 1 << 20;  // 1,048,576 iterations
            if (count > kMaxUnroll) {
                fail(n_tok, "range(N) unroll limit exceeded (max " +
                                std::to_string(kMaxUnroll) + ")");
            }
            return Stmt{ForStmt{std::move(var), count, std::move(body)}};
        }
        if (peek().kind == Tok::Ident && peek(1).kind == Tok::Equals) {
            std::string name = consume().text;
            consume(); // '='
            ExprPtr v = parse_expression();
            if (peek().kind == Tok::Newline) consume();
            return Stmt{AssignStmt{std::move(name), std::move(v)}};
        }
        fail(peek(), "expected 'return', 'if', 'for', or assignment");
    }

    // expr := term (cmp_op term)?
    ExprPtr parse_expression() {
        // Bound nested-parenthesis / method-chain / call-argument recursion:
        // parse_expression -> parse_term -> ... -> parse_unary -> '(' expr ')'
        // (and method/call args) re-enters parse_expression, so guarding here
        // caps the entire expression-recursion cycle on untrusted input.
        DepthGuard guard(*this);
        ExprPtr lhs = parse_term();
        if (peek().kind == Tok::Less || peek().kind == Tok::Greater) {
            char op = peek().kind == Tok::Less ? '<' : '>';
            consume();
            ExprPtr rhs = parse_term();
            return std::make_shared<Expr>(Expr{CmpOpExpr{op, lhs, rhs}});
        }
        return lhs;
    }

    ExprPtr parse_term() {
        ExprPtr lhs = parse_mul_term();
        while (peek().kind == Tok::Plus || peek().kind == Tok::Minus) {
            char op = peek().kind == Tok::Plus ? '+' : '-';
            consume();
            ExprPtr rhs = parse_mul_term();
            lhs = std::make_shared<Expr>(Expr{BinOpExpr{op, lhs, rhs}});
        }
        return lhs;
    }

    ExprPtr parse_mul_term() {
        ExprPtr lhs = parse_factor();
        while (peek().kind == Tok::Star || peek().kind == Tok::Slash) {
            char op = peek().kind == Tok::Star ? '*' : '/';
            consume();
            ExprPtr rhs = parse_factor();
            lhs = std::make_shared<Expr>(Expr{BinOpExpr{op, lhs, rhs}});
        }
        return lhs;
    }

    ExprPtr parse_factor() {
        ExprPtr e = parse_unary();
        while (peek().kind == Tok::Dot) {
            consume();
            const Token& method = expect(Tok::Ident, "expected method name after '.'");
            expect(Tok::LParen, "expected '(' after method name");
            std::vector<ExprPtr> args;
            if (peek().kind != Tok::RParen) {
                args.push_back(parse_expression());
                while (peek().kind == Tok::Comma) {
                    consume();
                    args.push_back(parse_expression());
                }
            }
            expect(Tok::RParen, "expected ')' to close method arguments");
            e = std::make_shared<Expr>(Expr{
                MethodCallExpr{e, method.text, std::move(args)}});
        }
        return e;
    }

    ExprPtr parse_unary() {
        const Token& t = peek();
        if (t.kind == Tok::Number) {
            consume();
            return std::make_shared<Expr>(Expr{NumberExpr{t.number}});
        }
        if (t.kind == Tok::Ident) {
            consume();
            return std::make_shared<Expr>(Expr{IdentExpr{t.text}});
        }
        if (t.kind == Tok::LParen) {
            consume();
            ExprPtr inner = parse_expression();
            expect(Tok::RParen, "expected ')' to close subexpression");
            return inner;
        }
        if (t.kind == Tok::Minus) {
            consume();
            ExprPtr operand = parse_factor();
            ExprPtr zero = std::make_shared<Expr>(Expr{NumberExpr{0.0}});
            return std::make_shared<Expr>(Expr{BinOpExpr{'-', zero, operand}});
        }
        fail(t, "expected number, identifier, '(', or '-' at start of expression");
    }
};

// ============================================================================
// Evaluator
// ============================================================================

using MethodFn = Variable (*)(const Variable&);

Variable call_relu(const Variable& v) {
    return Variable(tenzor::clamp(v.tensor(), 0.0f,
                                  std::numeric_limits<float>::infinity()),
                    v.requires_grad());
}
Variable call_sigmoid(const Variable& v) { return Variable(tenzor::sigmoid(v.tensor()), v.requires_grad()); }
Variable call_tanh(const Variable& v)    { return Variable(tenzor::tanh(v.tensor()),    v.requires_grad()); }
Variable call_exp(const Variable& v)     { return Variable(tenzor::exp(v.tensor()),     v.requires_grad()); }
Variable call_log(const Variable& v)     { return Variable(tenzor::log(v.tensor()),     v.requires_grad()); }
Variable call_sqrt(const Variable& v)    { return Variable(tenzor::sqrt(v.tensor()),    v.requires_grad()); }
Variable call_abs(const Variable& v)     { return Variable(tenzor::abs(v.tensor()),     v.requires_grad()); }
Variable call_neg(const Variable& v)     { return Variable(tenzor::neg(v.tensor()),     v.requires_grad()); }
Variable call_sum(const Variable& v)     { return Variable(tenzor::sum(v.tensor()),     v.requires_grad()); }
Variable call_mean(const Variable& v)    { return Variable(tenzor::mean(v.tensor()),    v.requires_grad()); }
Variable call_sin(const Variable& v)     { return Variable(tenzor::sin(v.tensor()),     v.requires_grad()); }
Variable call_cos(const Variable& v)     { return Variable(tenzor::cos(v.tensor()),     v.requires_grad()); }

const std::unordered_map<std::string, MethodFn>& method_table() {
    static const std::unordered_map<std::string, MethodFn> table = {
        {"relu",    call_relu},   {"sigmoid", call_sigmoid},
        {"tanh",    call_tanh},   {"exp",     call_exp},
        {"log",     call_log},    {"sqrt",    call_sqrt},
        {"abs",     call_abs},    {"neg",     call_neg},
        {"sum",     call_sum},    {"mean",    call_mean},
        {"sin",     call_sin},    {"cos",     call_cos},
    };
    return table;
}

using Env = std::unordered_map<std::string, Variable>;

class ScriptEvaluator {
public:
    ScriptEvaluator(std::vector<std::string> arg_names, std::vector<Stmt> body)
        : arg_names_(std::move(arg_names)), body_(std::move(body)) {}

    Variable evaluate(const std::vector<Variable>& args) const {
        if (args.size() != arg_names_.size()) {
            throw std::runtime_error(
                "compile_script: argument count mismatch — script expects " +
                std::to_string(arg_names_.size()) + ", got " +
                std::to_string(args.size()));
        }
        // Inherit scalar-literal device/dtype from the first input so
        // `x * 2.0 + 1.0` on a CUDA Float64 tensor emits CUDA Float64
        // scalars rather than CPU Float32 ones (which would trigger a
        // device-mismatch failure inside the op dispatch).
        Device ctx_device = Device::cpu();
        DType ctx_dtype = DType::Float32;
        if (!args.empty()) {
            ctx_device = args[0].tensor().device();
            ctx_dtype = args[0].tensor().dtype();
        }
        ctx_device_ = ctx_device;
        ctx_dtype_ = ctx_dtype;

        Env env;
        for (size_t i = 0; i < arg_names_.size(); ++i) {
            env.emplace(arg_names_[i], args[i]);
        }
        std::optional<Variable> returned;
        exec_block(body_, env, returned);
        if (!returned) {
            throw std::runtime_error("compile_script: execution completed without returning");
        }
        return *returned;
    }

private:
    std::vector<std::string> arg_names_;
    std::vector<Stmt> body_;
    mutable Device ctx_device_{Device::cpu()};
    mutable DType  ctx_dtype_{DType::Float32};

    // exec_block runs statements in order. Sets `returned` and stops on return.
    void exec_block(const std::vector<Stmt>& stmts, Env& env,
                    std::optional<Variable>& returned) const {
        for (const auto& s : stmts) {
            if (returned) return;
            exec_stmt(s, env, returned);
        }
    }

    void exec_stmt(const Stmt& s, Env& env, std::optional<Variable>& returned) const {
        std::visit([&](auto&& node) { exec_one(node, env, returned); }, s.node);
    }

    void exec_one(const ReturnStmt& r, Env& env, std::optional<Variable>& returned) const {
        returned = eval(*r.value, env);
    }

    void exec_one(const AssignStmt& a, Env& env, std::optional<Variable>&) const {
        env.insert_or_assign(a.target, eval(*a.value, env));
    }

    void exec_one(const IfStmt& iff, Env& env, std::optional<Variable>& returned) const {
        // Trace-based JIT semantic: a scripted `if` bakes in whichever branch is
        // taken at trace time. The condition is data-dependent (it is derived
        // from the traced tensors) and cannot be represented in the IR — the
        // comparison ops that produce it (lt/gt) have no OpType, and even a
        // representable scalar condition would need trace_if subgraph dispatch.
        // So reading the scalar to pick a branch is a GRAPH BREAK: under
        // TENZOR_JIT_STRICT it must throw (rather than silently freezing one
        // branch); otherwise it warns and the taken branch is recorded, which
        // matches eager for the traced configuration.
        //
        // Route the scalar read through Tensor::item(), which fires the
        // graph-break hook (see core/jit_hooks.hpp). A raw data<float>()[0]
        // bypasses the hook and would silently bake one branch even in strict
        // mode — the bug this replaces.
        Variable cond_var = eval(*iff.cond, env);
        Tensor cond_t = cond_var.tensor();
        // item<float>() requires a single-element Float32 tensor; the condition
        // inherits ctx_dtype_ so it may be Float64/Float16/BFloat16.
        if (cond_t.dtype() != DType::Float32) cond_t = cond_t.to(DType::Float32);
        bool cond_val = cond_t.item<float>() != 0.0f;
        const auto& body = cond_val ? iff.then_body : iff.else_body;
        exec_block(body, env, returned);
    }

    void exec_one(const ForStmt& f, Env& env, std::optional<Variable>& returned) const {
        // Static unrolling: iterate N times, each iteration sees i = 0..N-1
        // as a scalar Float32 Variable. The traced graph captures the full
        // unrolled sequence (matching Python semantics for fixed-N loops).
        for (int64_t i = 0; i < f.count; ++i) {
            if (returned) return;
            Tensor iv_t = full({}, static_cast<float>(i), DType::Float32, ctx_device_);
            if (ctx_dtype_ != DType::Float32) iv_t = iv_t.to(ctx_dtype_);
            Variable iv(iv_t, false);
            env.insert_or_assign(f.var, iv);
            exec_block(f.body, env, returned);
        }
    }

    Variable eval(const Expr& e, const Env& env) const {
        return std::visit([&](auto&& node) -> Variable { return visit(node, env); }, e.node);
    }

    Variable visit(const NumberExpr& n, const Env&) const {
        Tensor t = full({}, static_cast<float>(n.value), DType::Float32, ctx_device_);
        if (ctx_dtype_ != DType::Float32) t = t.to(ctx_dtype_);
        return Variable(t, false);
    }

    Variable visit(const IdentExpr& id, const Env& env) const {
        auto it = env.find(id.name);
        if (it == env.end()) {
            throw std::runtime_error(
                "compile_script: unknown identifier '" + id.name + "'");
        }
        return it->second;
    }

    Variable visit(const MethodCallExpr& m, const Env& env) const {
        Variable recv = eval(*m.receiver, env);

        // Handle argument-taking methods before falling through to the
        // zero-argument dispatch table.
        if (!m.args.empty()) {
            return call_method_with_args(recv, m.method, m.args, env);
        }

        const auto& table = method_table();
        auto it = table.find(m.method);
        if (it == table.end()) {
            throw std::runtime_error(
                "compile_script: unknown zero-argument method '." + m.method +
                "()' (supported: relu, sigmoid, tanh, exp, log, sqrt, abs, "
                "neg, sum, mean, sin, cos)");
        }
        return it->second(recv);
    }

    // Evaluate an expression that must reduce to a scalar Float32 value.
    float eval_scalar(const Expr& e, const Env& env) const {
        Variable v = eval(e, env);
        Tensor t = v.tensor();
        if (t.numel() != 1) {
            throw std::runtime_error(
                "compile_script: method argument must be a scalar");
        }
        if (t.dtype() != DType::Float32) t = t.to(DType::Float32);
        if (t.device().type != Device::Type::CPU) t = t.to(Device::cpu());
        return t.data<float>()[0];
    }

    // Evaluate an expression that must reduce to an integer.
    int64_t eval_int(const Expr& e, const Env& env) const {
        Variable v = eval(e, env);
        Tensor t = v.tensor();
        if (t.numel() != 1) {
            throw std::runtime_error(
                "compile_script: method argument must be a scalar integer");
        }
        if (t.device().type != Device::Type::CPU) t = t.to(Device::cpu());
        // A NumberExpr literal is stamped with the trace's context dtype (the
        // first input's dtype), so an integer method argument (e.g. `.sum(dim)`)
        // can arrive as ANY numeric dtype — Float16/BFloat16/Int*/Float64, not
        // just Float32/64. Normalize through Int64 (exact for any integer these
        // carry, truncating toward zero like the old static_cast) rather than
        // rejecting the other dtypes, which broke integer args for F16/BF16/int
        // traces.
        if (t.dtype() != DType::Int64) t = t.to(DType::Int64);
        return t.data<int64_t>()[0];
    }

    Variable call_method_with_args(const Variable& recv,
                                   const std::string& name,
                                   const std::vector<ExprPtr>& args,
                                   const Env& env) const {
        auto require_args = [&](size_t n, const char* sig) {
            if (args.size() != n) {
                throw std::runtime_error(
                    std::string("compile_script: method '.") + name +
                    "(" + sig + ")' expects " + std::to_string(n) +
                    " argument(s), got " + std::to_string(args.size()));
            }
        };

        if (name == "pow") {
            require_args(1, "exponent");
            float exp_val = eval_scalar(*args[0], env);
            return Variable(tenzor::pow(recv.tensor(), exp_val),
                            recv.requires_grad());
        }
        if (name == "clamp") {
            require_args(2, "min, max");
            float lo = eval_scalar(*args[0], env);
            float hi = eval_scalar(*args[1], env);
            return Variable(tenzor::clamp(recv.tensor(), lo, hi),
                            recv.requires_grad());
        }
        if (name == "sum" || name == "mean") {
            require_args(1, "dim");
            int64_t dim = eval_int(*args[0], env);
            Tensor out = (name == "sum") ? tenzor::sum(recv.tensor(), dim)
                                         : tenzor::mean(recv.tensor(), dim);
            return Variable(std::move(out), recv.requires_grad());
        }
        throw std::runtime_error(
            "compile_script: unknown method '." + name + "(...)' "
            "(methods with arguments: pow(exp), clamp(min, max), "
            "sum(dim), mean(dim))");
    }

    Variable visit(const BinOpExpr& b, const Env& env) const {
        Variable lhs = eval(*b.lhs, env);
        Variable rhs = eval(*b.rhs, env);
        auto normalise = [&](Variable& a, Variable& other) {
            if (a.tensor().numel() == 1 &&
                (a.tensor().dtype() != other.tensor().dtype() ||
                 a.tensor().device() != other.tensor().device())) {
                a = Variable(
                    a.tensor().to(other.tensor().device()).to(other.tensor().dtype()),
                    a.requires_grad());
            }
        };
        normalise(lhs, rhs);
        normalise(rhs, lhs);

        Tensor out;
        switch (b.op) {
            case '+': out = lhs.tensor() + rhs.tensor(); break;
            case '-': out = lhs.tensor() - rhs.tensor(); break;
            case '*': out = lhs.tensor() * rhs.tensor(); break;
            case '/': out = lhs.tensor() / rhs.tensor(); break;
            default:
                throw std::runtime_error(
                    std::string("compile_script: unknown operator '") + b.op + "'");
        }
        return Variable(out, lhs.requires_grad() || rhs.requires_grad());
    }

    Variable visit(const CmpOpExpr& c, const Env& env) const {
        Variable lhs = eval(*c.lhs, env);
        Variable rhs = eval(*c.rhs, env);
        Tensor out;
        if (c.op == '<')      out = tenzor::lt(lhs.tensor(), rhs.tensor());
        else                  out = tenzor::gt(lhs.tensor(), rhs.tensor());
        // Cast bool → float32 so jit::cond's `item<float>` call works on
        // scalar outputs.
        out = out.to(DType::Float32);
        return Variable(out, false);
    }
};

// ============================================================================
// Wrapper nn::Module for the tracer
// ============================================================================

class ScriptModule : public nn::Module {
public:
    ScriptModule(std::vector<std::string> arg_names, std::vector<Stmt> body)
        : evaluator_(std::move(arg_names), std::move(body)) {}

    Variable forward_impl(const Variable& x) override {
        return evaluator_.evaluate({x});
    }

    // Audit J6: needed by `compile_script_multi_with_dummies` to construct
    // the function-tracing closure. Returns a reference so the closure can
    // capture the evaluator without copying its body (which contains
    // unique_ptr-held AST nodes).
    auto evaluator() const -> const ScriptEvaluator& { return evaluator_; }

private:
    ScriptEvaluator evaluator_;
};

// ============================================================================
// Public entry point
// ============================================================================

// Hardening: cap the accepted script source length. compile_script forwards a
// raw (potentially untrusted) string into this hand-written lexer/parser; an
// arbitrarily large input is a DoS vector (memory + parse time) even before any
// nesting-depth concern. 1 MiB dwarfs any legitimate hand-written script.
static constexpr size_t kMaxScriptSourceBytes = 1u << 20;  // 1 MiB

static void validate_script_length(const char* source) {
    // strnlen-style bounded scan: stop at the cap so we never walk an
    // unterminated / huge buffer further than necessary.
    size_t len = 0;
    while (len <= kMaxScriptSourceBytes && source[len] != '\0') ++len;
    if (len > kMaxScriptSourceBytes) {
        throw std::runtime_error(
            "compile_script: source exceeds maximum supported length (" +
            std::to_string(kMaxScriptSourceBytes) + " bytes)");
    }
}

auto compile_script_with_dummy(const char* source, const Tensor& dummy)
    -> std::shared_ptr<CompiledModule> {
    if (source == nullptr) {
        throw std::runtime_error("compile_script: null source");
    }
    validate_script_length(source);

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(std::move(tokens));
    FuncDef fn = parser.parse_function();

    if (fn.args.size() != 1) {
        throw std::runtime_error(
            "compile_script(source, dummy): single-argument scripts only — "
            "use compile_script(source, std::vector<Tensor> dummies) for "
            "multi-argument scripts (got " +
            std::to_string(fn.args.size()) + " args).");
    }

    auto module = std::make_shared<ScriptModule>(fn.args, std::move(fn.body));
    auto module_base = std::static_pointer_cast<nn::Module>(module);

    auto compiled = jit::trace(module_base, dummy);
    // Remember the source module so forward() can retrace on shape / device /
    // dtype mismatch. The default `compile_script(source)` entry point uses a
    // CPU Float32 {1} dummy; without retrace, calling forward() with a CUDA or
    // Float64 input would run the CPU-baked graph and fail.
    compiled->set_source_module(module_base);
    return compiled;
}

// Audit J6: multi-argument compile_script. Parses an N-arg script, builds a
// ScriptModule with the same N-arg evaluator (which has always supported
// multi-arg internally — the gate was only at this top-level entry point),
// then traces via the function-based jit::trace overload which accepts a
// vector of input Variables. The resulting Graph is wrapped in a
// CompiledModule whose `forward(std::vector<Variable>)` overload (already
// declared on CompiledModule) is the user-facing entry point.
auto compile_script_multi_with_dummies(const char* source,
                                        const std::vector<Tensor>& dummies)
    -> std::shared_ptr<CompiledModule> {
    if (source == nullptr) {
        throw std::runtime_error("compile_script: null source");
    }
    validate_script_length(source);
    Lexer lexer(source);
    auto tokens = lexer.tokenize();
    Parser parser(std::move(tokens));
    FuncDef fn = parser.parse_function();

    if (fn.args.size() != dummies.size()) {
        throw std::runtime_error(
            "compile_script: script expects " + std::to_string(fn.args.size()) +
            " args but " + std::to_string(dummies.size()) + " dummies were provided");
    }

    auto module = std::make_shared<ScriptModule>(fn.args, std::move(fn.body));

    // Build input Variables from the dummy Tensors and invoke the function-
    // based trace. Each ScriptEvaluator call returns a single Variable; wrap
    // it in a 1-element vector for the closure's return type.
    std::vector<Variable> input_vars;
    input_vars.reserve(dummies.size());
    for (const auto& d : dummies) input_vars.emplace_back(d, /*requires_grad=*/false);

    auto closure = [module](const std::vector<Variable>& args) -> std::vector<Variable> {
        return {module->evaluator().evaluate(args)};
    };

    auto graph = jit::trace(closure, input_vars);
    if (!graph) {
        throw std::runtime_error("compile_script: jit::trace produced null graph");
    }
    auto compiled = std::make_shared<CompiledModule>(graph);
    auto module_base = std::static_pointer_cast<nn::Module>(module);
    compiled->set_source_module(module_base);
    return compiled;
}

} // namespace

auto compile_script(const char* source) -> std::shared_ptr<CompiledModule> {
    Tensor dummy = ones({1}, DType::Float32, Device::cpu());
    return compile_script_with_dummy(source, dummy);
}

auto compile_script(const char* source, const Tensor& dummy)
    -> std::shared_ptr<CompiledModule> {
    return compile_script_with_dummy(source, dummy);
}

// Audit J6: multi-arg public API.
auto compile_script(const char* source, const std::vector<Tensor>& dummies)
    -> std::shared_ptr<CompiledModule> {
    if (dummies.size() == 1) {
        return compile_script_with_dummy(source, dummies[0]);
    }
    return compile_script_multi_with_dummies(source, dummies);
}

} // namespace tenzor::jit
