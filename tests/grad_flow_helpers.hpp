#pragma once

// Layer-test helpers that assert gradient actually flowed through a Variable
// after backward(). Catches the silently-zeroed-gradient class of bug, where a
// raw-tensor op inside a layer's forward severs the grad_fn chain so backward
// fills .grad() with zeros instead of throwing.
//
// Use after calling .backward() in a layer/op test:
//
//   loss.backward();
//   EXPECT_GRAD_FLOWS(input);
//   EXPECT_GRAD_FLOWS(weight);
//
// EXPECT_GRAD_FLOWS asserts the Variable's grad has at least one non-zero
// element. EXPECT_GRAD_FLOWS_REL takes a tolerance for cases where genuine
// numerical noise dominates over the expected gradient signal.

#include <gtest/gtest.h>

#include "tenzor/autograd/variable.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"

#define EXPECT_GRAD_FLOWS(var)                                                 \
    do {                                                                       \
        const auto& _grad_opt = (var).grad();                                  \
        ASSERT_TRUE(_grad_opt.has_value())                                     \
            << "grad not defined for " #var                                    \
               " — backward likely never ran or grad_fn missing";              \
        ASSERT_GT(_grad_opt.value().numel(), 0)                                \
            << "grad for " #var " is empty (zero-element tensor)";             \
        /* Use .cpu() then .to(Float64) — the .to(Device) form has hit       \
         * oneapi-side dispatch issues ("expand: 'shape' attribute is        \
         * required") when transferring grads off the device. .cpu() is the  \
         * explicit, dedicated host-transfer path and works on every backend.*/\
        auto _g_cpu_f64 = _grad_opt.value().cpu().to(::tenzor::DType::Float64); \
        auto _g_max = ::tenzor::max(::tenzor::abs(_g_cpu_f64))                 \
                          .template item<double>();                            \
        EXPECT_GT(_g_max, 0.0)                                                 \
            << "grad_fn likely severed for " #var                              \
               " (grad tensor is identically zero after backward — "           \
               "raw-tensor op in forward path?)";                              \
    } while (0)

#define EXPECT_GRAD_FLOWS_REL(var, atol)                                       \
    do {                                                                       \
        const auto& _grad_opt = (var).grad();                                  \
        ASSERT_TRUE(_grad_opt.has_value())                                     \
            << "grad not defined for " #var;                                   \
        ASSERT_GT(_grad_opt.value().numel(), 0)                                \
            << "grad for " #var " is empty";                                   \
        /* Use .cpu() then .to(Float64) — the .to(Device) form has hit       \
         * oneapi-side dispatch issues ("expand: 'shape' attribute is        \
         * required") when transferring grads off the device. .cpu() is the  \
         * explicit, dedicated host-transfer path and works on every backend.*/\
        auto _g_cpu_f64 = _grad_opt.value().cpu().to(::tenzor::DType::Float64); \
        auto _g_max = ::tenzor::max(::tenzor::abs(_g_cpu_f64))                 \
                          .template item<double>();                            \
        EXPECT_GT(_g_max, (atol))                                              \
            << #var " grad max-abs " << _g_max << " below tolerance "          \
            << (atol);                                                         \
    } while (0)
