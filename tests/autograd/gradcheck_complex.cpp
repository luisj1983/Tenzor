/**
 * @file gradcheck_complex.cpp
 * @brief Implementation of gradcheck_complex.
 *
 * See gradcheck_complex.hpp for the high-level contract. The implementation
 * mirrors gradcheck()'s structure (per-element ± eps perturbation, central
 * differences) but keeps Re(y) and Im(y) as separate scalar contractions so
 * the full complex Jacobian is exercised.
 */
#include "gradcheck_complex.hpp"

#include <tenzor/autograd/ops.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/core/dtype.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>

#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tenzor_test {

using tenzor::Variable;
using tenzor::Tensor;
using tenzor::Device;
using tenzor::DType;
using tenzor::dtype_name;
using tenzor::zeros;

namespace {


// Sum (Re + i*Im) contractions of a complex output into a single complex
// scalar over CPU-resident data. The caller owns the tensor and is
// responsible for materialising it as contiguous CPU.
auto sum_complex(const tenzor::Tensor& t) -> std::complex<double> {
    double re = 0.0, im = 0.0;
    int64_t n = t.numel();
    if (t.dtype() == DType::Complex64) {
        const auto* p = t.data<std::complex<float>>();
        for (int64_t j = 0; j < n; ++j) {
            re += static_cast<double>(p[j].real());
            im += static_cast<double>(p[j].imag());
        }
    } else if (t.dtype() == DType::Complex128) {
        const auto* p = t.data<std::complex<double>>();
        for (int64_t j = 0; j < n; ++j) {
            re += p[j].real();
            im += p[j].imag();
        }
    } else {
        throw std::runtime_error(
            "gradcheck_complex: expected Complex64/Complex128 output, got "
            + std::string(dtype_name(t.dtype())));
    }
    return {re, im};
}

// Make a CPU, contiguous view (so data<T>()[i] indexes logical elements).
auto to_cpu_contig(const tenzor::Tensor& t) -> tenzor::Tensor {
    Tensor out = (t.device() == Device::cpu()) ? t : t.to(Device::cpu());
    if (!out.is_contiguous()) out = out.contiguous();
    return out;
}

// Run backward with a constant complex seed `seed_val` broadcast to the
// shape of `output`. Returns the input gradient as a real Tensor (which
// for a real input is the real-valued Jacobian-vector product).
auto run_backward_with_seed(
    std::function<tenzor::Variable(const tenzor::Variable&)> func,
    const tenzor::Variable& input,
    std::complex<double> seed_val) -> tenzor::Tensor {

    Variable input_copy(input.tensor().clone(), true);
    Variable output = func(input_copy);

    DType out_dt = output.tensor().dtype();
    Device dev = output.tensor().device();

    Tensor seed_cpu = zeros(
        std::vector<int64_t>(output.shape().begin(), output.shape().end()),
        out_dt, Device::cpu());
    int64_t n = seed_cpu.numel();
    if (out_dt == DType::Complex64) {
        auto* p = seed_cpu.data<std::complex<float>>();
        std::complex<float> sv{static_cast<float>(seed_val.real()),
                               static_cast<float>(seed_val.imag())};
        for (int64_t j = 0; j < n; ++j) p[j] = sv;
    } else if (out_dt == DType::Complex128) {
        auto* p = seed_cpu.data<std::complex<double>>();
        for (int64_t j = 0; j < n; ++j) p[j] = seed_val;
    } else {
        throw std::runtime_error(
            "gradcheck_complex: backward expects complex output");
    }
    Tensor seed = (dev == Device::cpu()) ? seed_cpu : seed_cpu.to(dev);

    output.backward(seed);
    if (!input_copy.has_grad()) {
        throw std::runtime_error(
            "gradcheck_complex: no gradient produced for input");
    }
    return *input_copy.grad();
}

}  // namespace

auto gradcheck_complex(
    std::function<tenzor::Variable(const tenzor::Variable&)> func,
    const tenzor::Variable& input,
    double eps,
    double atol,
    double rtol) -> bool {

    if (!input.requires_grad()) {
        std::cerr << "gradcheck_complex: input must have requires_grad=true\n";
        return false;
    }
    if (input.dtype() == DType::Float32 && eps < 5e-4) {
        eps = 5e-4;  // avoid catastrophic cancellation
    }

    // ----- Numerical Jacobian: split into Re and Im accumulators per input. ---
    int64_t total = input.tensor().numel();
    Tensor x = input.tensor().clone();
    if (x.device() != Device::cpu()) x = x.to(Device::cpu());
    if (!x.is_contiguous()) x = x.contiguous();
    Device orig_dev = input.tensor().device();

    std::vector<double> num_re(static_cast<size_t>(total), 0.0);
    std::vector<double> num_im(static_cast<size_t>(total), 0.0);

    for (int64_t i = 0; i < total; ++i) {
        Tensor x_plus = x.clone();
        Tensor x_minus = x.clone();
        if (input.dtype() == DType::Float32) {
            x_plus.data<float>()[i] += static_cast<float>(eps);
            x_minus.data<float>()[i] -= static_cast<float>(eps);
        } else if (input.dtype() == DType::Float64) {
            x_plus.data<double>()[i] += eps;
            x_minus.data<double>()[i] -= eps;
        } else {
            std::cerr << "gradcheck_complex: only Float32/Float64 real input supported\n";
            return false;
        }
        Variable v_plus = (orig_dev == Device::cpu())
                              ? Variable(x_plus, false)
                              : Variable(x_plus.to(orig_dev), false);
        Variable v_minus = (orig_dev == Device::cpu())
                               ? Variable(x_minus, false)
                               : Variable(x_minus.to(orig_dev), false);

        Tensor y_plus = to_cpu_contig(func(v_plus).tensor());
        Tensor y_minus = to_cpu_contig(func(v_minus).tensor());

        auto s_plus = sum_complex(y_plus);
        auto s_minus = sum_complex(y_minus);
        auto diff = (s_plus - s_minus) / (2.0 * eps);
        num_re[static_cast<size_t>(i)] = diff.real();
        num_im[static_cast<size_t>(i)] = diff.imag();
    }

    // ----- Analytical: two backward passes with Re-only and Im-only seeds. ---
    Tensor ana_re_t = to_cpu_contig(run_backward_with_seed(func, input, {1.0, 0.0}));
    Tensor ana_im_t = to_cpu_contig(run_backward_with_seed(func, input, {0.0, 1.0}));

    auto read_real = [&](const Tensor& t, int64_t i) -> double {
        if (t.dtype() == DType::Float32) return t.data<float>()[i];
        if (t.dtype() == DType::Float64) return t.data<double>()[i];
        throw std::runtime_error(
            "gradcheck_complex: input grad must be real, got "
            + std::string(dtype_name(t.dtype())));
    };

    bool ok = true;
    double max_abs_re = 0.0, max_abs_im = 0.0;
    int64_t fails = 0;
    for (int64_t i = 0; i < total; ++i) {
        double a_re = read_real(ana_re_t, i);
        double a_im = read_real(ana_im_t, i);
        double n_re = num_re[static_cast<size_t>(i)];
        double n_im = num_im[static_cast<size_t>(i)];

        double e_re = std::abs(a_re - n_re);
        double e_im = std::abs(a_im - n_im);
        max_abs_re = std::max(max_abs_re, e_re);
        max_abs_im = std::max(max_abs_im, e_im);

        double thr_re = atol + rtol * std::abs(a_re);
        double thr_im = atol + rtol * std::abs(a_im);
        if (e_re > thr_re || e_im > thr_im) {
            if (fails < 5) {
                std::cerr << "gradcheck_complex[" << i << "]:"
                          << " ana=(" << a_re << "," << a_im << ")"
                          << " num=(" << n_re << "," << n_im << ")"
                          << " err=(" << e_re << "," << e_im << ")\n";
            }
            ++fails;
            ok = false;
        }
    }
    if (!ok) {
        std::cerr << "gradcheck_complex: " << fails << "/" << total
                  << " elements failed; max_abs_re=" << max_abs_re
                  << " max_abs_im=" << max_abs_im
                  << " (atol=" << atol << " rtol=" << rtol << ")\n";
    }
    return ok;
}

}  // namespace tenzor_test
