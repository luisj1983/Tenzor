/**
 * @file script.cpp
 * @brief Minimal Python-subset script compiler (see include/tenzor/jit/script.hpp).
 *
 * Implementation strategy: parse the script into an argument list and an
 * expression AST, wrap that in a small `nn::Module` subclass whose
 * `forward_impl` evaluates the AST using Variable arithmetic, then hand the
 * module to `jit::trace` — which already takes care of Graph construction,
 * operation recording, and CompiledModule packaging.
 *
 * This keeps the scripting frontend's surface area to only what the language
 * subset requires (lexer + recursive-descent parser + tiny interpreter) while
 * reusing the full, tested tracer/compiler machinery for execution.
 */

#include <tenzor/jit/script.hpp>

#include <cctype>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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
    Def, Ident, Number, Return,
    LParen, RParen, Comma, Colon,
    Plus, Minus, Star, Slash,
    End
};

struct Token {
    Tok kind;
    std::string text;   // Ident name or number literal
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
    explicit Lexer(const char* src) : src_(src) {}

    std::vector<Token> tokenize() {
        std::vector<Token> out;
        while (*src_) {
            skip_spaces_and_comments();
            if (!*src_) break;
            int start_line = line_, start_col = col_;
            char c = *src_;
            if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                out.push_back(read_ident(start_line, start_col));
            } else if (std::isdigit(static_cast<unsigned char>(c)) ||
                       (c == '.' && std::isdigit(static_cast<unsigned char>(peek(1))))) {
                out.push_back(read_number(start_line, start_col));
            } else {
                out.push_back(read_punct(start_line, start_col));
            }
        }
        out.push_back({Tok::End, "<eof>", 0.0, line_, col_});
        return out;
    }

private:
    const char* src_;
    int line_ = 1;
    int col_ = 1;

    char peek(int offset) const { return src_[offset]; }

    void advance() {
        if (*src_ == '\n') { ++line_; col_ = 1; } else { ++col_; }
        ++src_;
    }

    void skip_spaces_and_comments() {
        while (*src_) {
            if (std::isspace(static_cast<unsigned char>(*src_))) {
                advance();
            } else if (*src_ == '#') {
                // Python-style line comment
                while (*src_ && *src_ != '\n') advance();
            } else {
                break;
            }
        }
    }

    Token read_ident(int start_line, int start_col) {
        std::string name;
        while (*src_ && (std::isalnum(static_cast<unsigned char>(*src_)) || *src_ == '_')) {
            name += *src_;
            advance();
        }
        Tok kind = Tok::Ident;
        if (name == "def") kind = Tok::Def;
        else if (name == "return") kind = Tok::Return;
        return {kind, name, 0.0, start_line, start_col};
    }

    Token read_number(int start_line, int start_col) {
        std::string lit;
        while (*src_ && (std::isdigit(static_cast<unsigned char>(*src_)) || *src_ == '.')) {
            lit += *src_;
            advance();
        }
        // Optional exponent: e.g. 1e5, 1.2e-3
        if (*src_ == 'e' || *src_ == 'E') {
            lit += *src_;
            advance();
            if (*src_ == '+' || *src_ == '-') { lit += *src_; advance(); }
            while (*src_ && std::isdigit(static_cast<unsigned char>(*src_))) {
                lit += *src_;
                advance();
            }
        }
        Token t{Tok::Number, lit, 0.0, start_line, start_col};
        t.number = std::strtod(lit.c_str(), nullptr);
        return t;
    }

    Token read_punct(int start_line, int start_col) {
        char c = *src_;
        advance();
        std::string s(1, c);
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
            default: {
                Token bad{Tok::End, s, 0.0, start_line, start_col};
                fail(bad, std::string("unexpected character '") + c + "'");
            }
        }
        return {kind, s, 0.0, start_line, start_col};
    }
};

// ============================================================================
// AST
// ============================================================================

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct NumberExpr { double value; };
struct IdentExpr  { std::string name; };
struct BinOpExpr  { char op; ExprPtr lhs; ExprPtr rhs; };

struct Expr {
    std::variant<NumberExpr, IdentExpr, BinOpExpr> node;
};

struct FuncDef {
    std::string name;
    std::vector<std::string> args;
    ExprPtr body;  // return expression
};

// ============================================================================
// Parser (recursive descent)
// ============================================================================

class Parser {
public:
    explicit Parser(std::vector<Token> tokens) : toks_(std::move(tokens)) {}

    FuncDef parse_function() {
        FuncDef f;
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
        expect(Tok::Return, "only 'return EXPR' is supported in the MVP grammar");
        f.body = parse_expression();
        // Trailing tokens are allowed (e.g. trailing newline handled by lexer);
        // anything non-trivial is an error.
        if (peek().kind != Tok::End) {
            fail(peek(), "unexpected token after 'return' expression");
        }
        return f;
    }

private:
    std::vector<Token> toks_;
    size_t pos_ = 0;

    const Token& peek() const { return toks_[pos_]; }
    const Token& consume() { return toks_[pos_++]; }

    const Token& expect(Tok kind, std::string_view msg) {
        if (peek().kind != kind) fail(peek(), msg);
        return consume();
    }

    // EXPR     := TERM (('+' | '-') TERM)*
    // TERM     := FACTOR (('*' | '/') FACTOR)*
    // FACTOR   := NUMBER | IDENT | '(' EXPR ')' | '-' FACTOR

    ExprPtr parse_expression() {
        ExprPtr lhs = parse_term();
        while (peek().kind == Tok::Plus || peek().kind == Tok::Minus) {
            char op = peek().kind == Tok::Plus ? '+' : '-';
            consume();
            ExprPtr rhs = parse_term();
            lhs = std::make_shared<Expr>(Expr{BinOpExpr{op, lhs, rhs}});
        }
        return lhs;
    }

    ExprPtr parse_term() {
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
            // Unary minus: parse as (0 - factor) — keeps codegen trivial.
            consume();
            ExprPtr operand = parse_factor();
            ExprPtr zero = std::make_shared<Expr>(Expr{NumberExpr{0.0}});
            return std::make_shared<Expr>(Expr{BinOpExpr{'-', zero, operand}});
        }
        fail(t, "expected number, identifier, '(', or '-' at start of expression");
    }
};

// ============================================================================
// Evaluator — walks the AST producing a Variable
// ============================================================================

class ScriptEvaluator {
public:
    // `args[i]` must correspond to `arg_names_[i]`.
    ScriptEvaluator(std::vector<std::string> arg_names, ExprPtr body)
        : arg_names_(std::move(arg_names)), body_(std::move(body)) {}

    Variable evaluate(const std::vector<Variable>& args) const {
        if (args.size() != arg_names_.size()) {
            throw std::runtime_error(
                "compile_script: argument count mismatch — script expects " +
                std::to_string(arg_names_.size()) + ", got " +
                std::to_string(args.size()));
        }
        return eval(*body_, args);
    }

private:
    std::vector<std::string> arg_names_;
    ExprPtr body_;

    Variable eval(const Expr& e, const std::vector<Variable>& args) const {
        return std::visit([&](auto&& node) -> Variable { return visit(node, args); }, e.node);
    }

    // A scalar literal is materialised as a Tensor in whichever dtype/device
    // the first encountered Variable lives on; if we only see literals before
    // an ident, we default to Float32/CPU and rely on broadcasting.
    Variable visit(const NumberExpr& n, const std::vector<Variable>&) const {
        Tensor t = full({}, static_cast<float>(n.value), DType::Float32, Device::cpu());
        return Variable(t, false);
    }

    Variable visit(const IdentExpr& id, const std::vector<Variable>& args) const {
        for (size_t i = 0; i < arg_names_.size(); ++i) {
            if (arg_names_[i] == id.name) return args[i];
        }
        throw std::runtime_error(
            "compile_script: unknown identifier '" + id.name + "'");
    }

    Variable visit(const BinOpExpr& b, const std::vector<Variable>& args) const {
        Variable lhs = eval(*b.lhs, args);
        Variable rhs = eval(*b.rhs, args);

        // Promote scalar literals to the operand's dtype/device when needed.
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
};

// ============================================================================
// Wrapper nn::Module that the tracer can consume
// ============================================================================

class ScriptModule : public nn::Module {
public:
    ScriptModule(std::vector<std::string> arg_names, ExprPtr body)
        : evaluator_(std::move(arg_names), std::move(body)) {}

    Variable forward_impl(const Variable& x) override {
        // Single-arg path — mirrors the only forward() overload that
        // CompiledModule exposes for the Variable type.
        return evaluator_.evaluate({x});
    }

private:
    ScriptEvaluator evaluator_;
};

} // namespace

// ============================================================================
// Public entry point
// ============================================================================

auto compile_script(const char* source) -> std::shared_ptr<CompiledModule> {
    if (source == nullptr) {
        throw std::runtime_error("compile_script: null source");
    }

    Lexer lexer(source);
    auto tokens = lexer.tokenize();

    Parser parser(std::move(tokens));
    FuncDef fn = parser.parse_function();

    if (fn.args.size() != 1) {
        // Multi-arg scripts require the vector-input CompiledModule::forward
        // overload. The MVP only supports the single-Variable path.
        throw std::runtime_error(
            "compile_script: MVP supports single-argument functions only "
            "(got " + std::to_string(fn.args.size()) + " args)");
    }

    auto module = std::make_shared<ScriptModule>(fn.args, fn.body);

    // Synthesize a dummy input shaped to trigger scalar broadcasting — actual
    // shapes come from the caller's later forward() call. {1} is enough for
    // elementwise arithmetic to record a representative graph.
    Tensor dummy = ones({1}, DType::Float32, Device::cpu());

    return tenzor::jit::trace(
        std::static_pointer_cast<nn::Module>(module), dummy);
}

} // namespace tenzor::jit
