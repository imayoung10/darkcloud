#include <torch/extension.h>
#include <cuda_runtime.h>

// CPU 텐서를 Unified Memory로 복사
torch::Tensor to_unified(torch::Tensor cpu_tensor) {
    TORCH_CHECK(!cpu_tensor.is_cuda(), "Input must be a CPU tensor");

    auto sizes    = cpu_tensor.sizes();
    auto dtype    = cpu_tensor.dtype();
    auto numel    = cpu_tensor.numel();
    auto elem_size = cpu_tensor.element_size();

    // Unified Memory 할당
    void* managed_ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&managed_ptr, numel * elem_size);
    TORCH_CHECK(err == cudaSuccess, "cudaMallocManaged failed");

    // CPU → Unified Memory 복사
    err = cudaMemcpy(managed_ptr,
                     cpu_tensor.data_ptr(),
                     numel * elem_size,
                     cudaMemcpyHostToDevice);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy failed");

    // TensorOptions: CUDA 디바이스로 생성
    auto options = torch::TensorOptions()
                       .dtype(dtype)
                       .device(torch::kCUDA)
                       .requires_grad(cpu_tensor.requires_grad());

    // from_blob으로 래핑, 해제 시 cudaFree 호출
    return torch::from_blob(
        managed_ptr,
        sizes,
        [managed_ptr](void* /*unused*/) {
            cudaFree(managed_ptr);
        },
        options
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("to_unified", &to_managed, "Copy CPU tensor to CUDA Unified Memory");
}
