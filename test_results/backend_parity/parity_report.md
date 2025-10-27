# Backend Parity Test Report
**Generated**: Sat 25 Oct 09:37:52 BST 2025
**Backends Tested**: cpu cuda vulkan oneapi

## Test Results

| Test Suite | CPU | CUDA | Vulkan | OneAPI |
|------------|-----|------|--------|--------|
| tenzor_unit_tests | ❌ FAIL | ❌ FAIL | ❌ FAIL | ❌ FAIL |
| tenzor_integration_tests | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| test_ciou_loss | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| test_slice_backend_parity | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |
| test_phase11_backends | ✅ PASS | ✅ PASS | ✅ PASS | ✅ PASS |

## Legend
- ✅ PASS: All tests passed
- ❌ FAIL: One or more tests failed
- ⏱️ TIMEOUT: Tests exceeded 300 second timeout
- ⏭️ SKIP: Backend not available or test not found

## Detailed Logs
See individual log files in: `/home/lee/Projects/Tenzor/test_results/backend_parity/`

## Backend Parity Analysis

### Full Parity (all backends pass)
- ✅ tenzor_integration_tests
- ✅ test_ciou_loss
- ✅ test_slice_backend_parity
- ✅ test_phase11_backends

### Partial Parity (some backends pass)
