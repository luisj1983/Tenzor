# Phase 12 Completion Action Plan

**Current Status:** 65% Complete
**Target Status:** 100% Complete
**Estimated Time:** 13.5 hours (Priority 1 only) or 20.5 hours (full completion)

---

## Immediate Actions (Priority 1)

### Action 1: Implement QuantizedConv2d::from_float()

**File:** `/home/lee/Projects/Tenzor/src/nn/quantization/quantized_layers.cpp:217-222`

**Current Code:**
```cpp
auto QuantizedConv2d::from_float(const Conv2d& fp_conv, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConv2d> {
    // Similar to Linear - quantize weights and create quantized layer
    // Implementation details omitted for brevity
    throw std::runtime_error("Not implemented - would quantize Conv2d weights");
}
```

**Required Implementation:**
```cpp
auto QuantizedConv2d::from_float(const Conv2d& fp_conv, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConv2d> {
    // Extract Conv2d parameters
    auto weight = fp_conv.weight();
    auto bias = fp_conv.bias();

    // Get quantization parameters from qconfig
    auto weight_observer = qconfig.create_weight_observer();

    // Observe weights to compute quantization parameters
    weight_observer->observe(weight.tensor());
    auto weight_qparams = weight_observer->calculate_qparams();

    // Quantize weights (per-channel for Conv2d)
    auto q_weight = quantize_per_channel(
        weight.tensor(),
        weight_qparams,
        /*channel_axis=*/0  // Output channels
    );

    // Create quantized layer
    auto q_conv = std::make_shared<QuantizedConv2d>(
        fp_conv.in_channels(),
        fp_conv.out_channels(),
        fp_conv.kernel_size(),
        weight_qparams,
        fp_conv.stride(),
        fp_conv.padding(),
        fp_conv.dilation(),
        fp_conv.groups()
    );

    // Set quantized weights
    q_conv->set_weight(q_weight);

    // Set bias (keep as FP32 or quantize with different params)
    if (bias.has_value()) {
        q_conv->set_bias(bias.value().tensor());
    }

    return q_conv;
}
```

**Dependencies:**
- Ensure `quantize_per_channel()` function exists in `quantize.cpp`
- Verify `QuantizedConv2d` constructor accepts these parameters
- Add proper error handling

**Time Estimate:** 6 hours
- Implementation: 4 hours
- Testing: 1 hour
- Debugging: 1 hour

---

### Action 2: Implement QuantizedBatchNorm2d::from_float()

**File:** `/home/lee/Projects/Tenzor/src/nn/quantization/quantized_layers.cpp:251-257`

**Current Code:**
```cpp
auto QuantizedBatchNorm2d::from_float(const Module& fp_bn, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedBatchNorm2d> {
    // Extract BN parameters and fold
    // Implementation would extract gamma, beta, running_mean, running_var
    // and compute folded scale = gamma / sqrt(var + eps), bias = beta - scale * mean
    throw std::runtime_error("Not implemented - would fold BN parameters");
}
```

**Required Implementation:**
```cpp
auto QuantizedBatchNorm2d::from_float(const Module& fp_bn, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedBatchNorm2d> {
    // Cast to BatchNorm2d
    auto& bn = dynamic_cast<const nn::BatchNorm2d&>(fp_bn);

    // Extract BatchNorm parameters
    auto gamma = bn.weight();      // scale parameter
    auto beta = bn.bias();         // shift parameter
    auto running_mean = bn.running_mean();
    auto running_var = bn.running_var();
    float eps = bn.eps();

    // Fold BatchNorm into affine transformation
    // scale = gamma / sqrt(var + eps)
    auto inv_std = (running_var + eps).sqrt().reciprocal();
    auto folded_scale = gamma.tensor() * inv_std;

    // bias = beta - scale * mean
    auto folded_bias = beta.tensor() - folded_scale * running_mean;

    // Create quantized BatchNorm (just an affine transform now)
    auto q_bn = std::make_shared<QuantizedBatchNorm2d>(
        bn.num_features(),
        folded_scale,
        folded_bias
    );

    return q_bn;
}
```

**Dependencies:**
- Access to `BatchNorm2d::running_mean()`, `running_var()`, `eps()`
- May need to add getters to `BatchNorm2d` class
- Verify folding math is correct

**Time Estimate:** 4 hours
- Implementation: 2 hours
- Adding getters to BatchNorm2d: 1 hour
- Testing: 1 hour

---

### Action 3: Implement QuantizedConv2dReLU::from_float()

**File:** `/home/lee/Projects/Tenzor/src/nn/quantization/quantized_layers.cpp:310-314`

**Current Code:**
```cpp
auto QuantizedConv2dReLU::from_float(const Conv2d& fp_conv, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConv2dReLU> {
    // Similar to QuantizedConv2d::from_float
    throw std::runtime_error("Not implemented");
}
```

**Required Implementation:**
```cpp
auto QuantizedConv2dReLU::from_float(const Conv2d& fp_conv, const QConfig& qconfig)
    -> std::shared_ptr<QuantizedConv2dReLU> {
    // Reuse QuantizedConv2d logic
    auto q_conv = QuantizedConv2d::from_float(fp_conv, qconfig);

    // Extract parameters from quantized conv
    auto q_weight = q_conv->weight();
    auto q_bias = q_conv->bias();

    // Create fused Conv2d+ReLU
    auto q_conv_relu = std::make_shared<QuantizedConv2dReLU>(
        fp_conv.in_channels(),
        fp_conv.out_channels(),
        fp_conv.kernel_size(),
        q_weight.params(),
        fp_conv.stride(),
        fp_conv.padding(),
        fp_conv.dilation(),
        fp_conv.groups()
    );

    // Set weights and bias
    q_conv_relu->set_weight(q_weight);
    if (q_bias.has_value()) {
        q_conv_relu->set_bias(q_bias.value());
    }

    return q_conv_relu;
}
```

**Dependencies:**
- `QuantizedConv2d::from_float()` must be implemented first
- Access to quantized weight/bias from `QuantizedConv2d`
- May need to add getters

**Time Estimate:** 2 hours
- Implementation: 1 hour
- Testing: 30 minutes
- Integration: 30 minutes

---

### Action 4: Fix Phase 12 Test Compilation

**File:** `/home/lee/Projects/Tenzor/include/tenzor/nn/quantization/qconfig.hpp:31-50`

**Option 1: Add Default Constructor (Recommended)**

Add to `QConfig` class:
```cpp
class QConfig {
public:
    // Default constructor
    QConfig()
        : QConfig(
            []() { return std::make_unique<MinMaxObserver>(); },
            []() { return std::make_unique<MinMaxObserver>(); }
          ) {}

    // Existing parameterized constructor
    QConfig(
        std::function<std::unique_ptr<Observer>()> weight_observer_factory,
        std::function<std::unique_ptr<Observer>()> activation_observer_factory,
        QuantDType weight_dtype = QuantDType::INT8,
        QuantDType activation_dtype = QuantDType::INT8,
        QuantizationScheme weight_scheme = QuantizationScheme::PerChannelSymmetric,
        QuantizationScheme activation_scheme = QuantizationScheme::PerTensorSymmetric
    );

    // ... rest of class
};
```

**Alternative: Fix Test Fixture**

Modify `/home/lee/Projects/Tenzor/tests/test_quantization_conversion.cpp:36-45`:
```cpp
class QuantizationConversionTest : public ::testing::Test {
protected:
    QuantizationConversionTest()
        : device_(Device::cpu()),
          qconfig_(DefaultQConfigs::default_qconfig()) {}

    Device device_;
    QConfig qconfig_;  // Now initialized in constructor
};
```

**Time Estimate:** 30 minutes
- Choose approach: 5 minutes
- Implement: 10 minutes
- Rebuild tests: 10 minutes
- Verify: 5 minutes

---

### Action 5: Run and Validate Phase 12 Tests

**Commands:**
```bash
cd /home/lee/Projects/Tenzor/build
cmake --build . --target test_quantization_conversion -j$(nproc)
cmake --build . --target test_mask_rcnn_losses -j$(nproc)
cmake --build . --target test_vulkan_complete_ops -j$(nproc)

cd /home/lee/Projects/Tenzor/bin
./test_quantization_conversion --gtest_output=xml:phase12_quantization.xml
./test_mask_rcnn_losses --gtest_output=xml:phase12_mask_rcnn.xml
./test_vulkan_complete_ops --gtest_output=xml:phase12_vulkan.xml
```

**Expected Results:**
- All tests compile successfully
- All tests pass
- No crashes or memory leaks

**Time Estimate:** 1 hour
- Build: 15 minutes
- Run tests: 15 minutes
- Analyze results: 15 minutes
- Fix any issues: 15 minutes

---

## Quality Assurance Actions (Priority 2)

### Action 6: Code Coverage Analysis

**Commands:**
```bash
cd /home/lee/Projects/Tenzor/build
cmake .. -DCMAKE_BUILD_TYPE=Coverage -DCMAKE_CXX_FLAGS="--coverage"
cmake --build . -j$(nproc)
ctest
lcov --capture --directory . --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info
lcov --list coverage.info
genhtml coverage.info --output-directory coverage_html
```

**Success Criteria:**
- Overall coverage ≥ 95%
- Core modules ≥ 98%
- Backends ≥ 90%
- Quantization ≥ 95%

**Time Estimate:** 1 hour

---

### Action 7: AddressSanitizer

**Commands:**
```bash
cd /home/lee/Projects/Tenzor/build
cmake .. -DCMAKE_CXX_FLAGS="-fsanitize=address -g"
cmake --build . -j$(nproc)
ctest --output-on-failure
```

**Success Criteria:**
- Zero memory leaks
- Zero use-after-free
- Zero buffer overflows

**Time Estimate:** 2 hours

---

### Action 8: ThreadSanitizer

**Commands:**
```bash
cd /home/lee/Projects/Tenzor/build
cmake .. -DCMAKE_CXX_FLAGS="-fsanitize=thread -g"
cmake --build . -j$(nproc)
ctest --output-on-failure
```

**Success Criteria:**
- Zero data races
- Zero deadlocks

**Time Estimate:** 2 hours

---

### Action 9: Static Analysis

**Commands:**
```bash
cd /home/lee/Projects/Tenzor
clang-tidy src/**/*.cpp include/**/*.hpp \
    -checks='*,-modernize-use-trailing-return-type' \
    -- -Iinclude -std=c++23

cppcheck --enable=all --inconclusive --std=c++23 \
    --suppress=missingInclude src/ include/
```

**Success Criteria:**
- Zero critical issues
- < 10 warnings
- All warnings documented

**Time Estimate:** 2 hours

---

## Timeline

### Day 1 (8 hours)
- [ ] Action 1: Implement QuantizedConv2d::from_float() (6 hours)
- [ ] Action 4: Fix test compilation (30 minutes)
- [ ] Action 2: Start QuantizedBatchNorm2d::from_float() (1.5 hours)

### Day 2 (5.5 hours)
- [ ] Action 2: Complete QuantizedBatchNorm2d::from_float() (2.5 hours)
- [ ] Action 3: Implement QuantizedConv2dReLU::from_float() (2 hours)
- [ ] Action 5: Run and validate Phase 12 tests (1 hour)

**Total Priority 1: 13.5 hours over 1.5 days**

### Day 3 (Optional - Quality Assurance)
- [ ] Action 6: Code coverage (1 hour)
- [ ] Action 7: AddressSanitizer (2 hours)
- [ ] Action 8: ThreadSanitizer (2 hours)
- [ ] Action 9: Static analysis (2 hours)

**Total Priority 2: 7 hours (1 day)**

---

## Success Criteria

**Phase 12 is COMPLETE when:**

✅ All 3 quantization conversion functions implemented
✅ All Phase 12 tests compile
✅ All Phase 12 tests pass
✅ Zero production code stubs remain
✅ Code coverage ≥ 95%
✅ All sanitizers pass
✅ Static analysis clean

---

## Risk Mitigation

**Risk 1: Implementation complexity higher than estimated**
- Mitigation: Start with simplest function (QuantizedConv2dReLU)
- Fallback: Request extension if > 20 hours required

**Risk 2: Missing dependencies (e.g., quantize_per_channel)**
- Mitigation: Implement missing utility functions as needed
- Estimate: +2 hours per missing function

**Risk 3: Test failures after implementation**
- Mitigation: Use existing quantization tests as reference
- Debugging time: Already budgeted (1 hour per function)

**Risk 4: Breaking changes to existing tests**
- Mitigation: Run full test suite after each implementation
- Rollback: Use git to revert if needed

---

## Next Steps

1. **Review this action plan** (you are here)
2. **Allocate development time** (13.5 hours for Priority 1)
3. **Execute actions in order** (follow timeline)
4. **Validate completion** (run all tests)
5. **Update Phase 12 status** (mark as COMPLETE)

---

**Generated:** 2025-10-24 13:52 UTC
**For Questions:** See PHASE_12_VALIDATION_REPORT.md
