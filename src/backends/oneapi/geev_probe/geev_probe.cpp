#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
int main() {
    sycl::queue q;
    (void)::oneapi::mkl::lapack::geev_scratchpad_size<float>(
        q, ::oneapi::mkl::job::vec, ::oneapi::mkl::job::novec, 1, 1);
    return 0;
}
