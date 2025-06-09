#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Custom deleter for CUDA Unified Memory
struct CUDAManagedDeleter {
    void operator()(void* ptr) {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                // Log error but don't throw as this is a destructor
                std::cerr << "Error freeing CUDA memory: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }
};

torch::Tensor make_um_tensor(int64_t num_elements) {
    void* ptr = nullptr;
    size_t size = num_elements * sizeof(float);

    // 1. Unified Memory 할당
    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocManaged failed");
    }

    // 2. PyTorch Tensor 래핑
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);  // device='cpu' for UMA
    auto tensor = torch::from_blob(ptr, {num_elements}, options);

    return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("make_um_tensor", &make_um_tensor, "Create Unified Memory Tensor (cudaMallocManaged)");
}
