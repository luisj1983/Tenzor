#include <cuda_runtime.h>
#include <iostream>

__global__ void write_scalar(float* output) {
    output[0] = 42.0f;
    printf("[CUDA] Wrote 42.0 to output[0] at address %p\n", (void*)output);
}

int main() {
    float* d_data;
    float h_result;

    // Allocate 4 bytes (one float) on device
    cudaMalloc(&d_data, sizeof(float));
    std::cout << "Allocated device memory at: " << (void*)d_data << std::endl;

    // Initialize to known garbage value
    float init_val = -999.0f;
    cudaMemcpy(d_data, &init_val, sizeof(float), cudaMemcpyHostToDevice);

    // Read back to verify initialization
    cudaMemcpy(&h_result, d_data, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "After init: " << h_result << std::endl;

    // Launch kernel to write 42.0
    write_scalar<<<1, 1>>>(d_data);
    cudaDeviceSynchronize();

    // Read back
    cudaMemcpy(&h_result, d_data, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "After kernel: " << h_result << " (expected 42.0)" << std::endl;

    cudaFree(d_data);
    return 0;
}
