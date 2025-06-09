import torch
import um_tensor
import ctypes

# 1. Unified Memory 기반 텐서 생성
x = um_tensor.make_um_tensor(1000000)  # 1 million floats

# 2. 이건 진짜로 cudaMallocManaged 기반이야!
print(x.shape)       # torch.Size([1000000])
print(x.device)      # cpu (UMA 기반)

# 3. Prefetch: C pointer로 전달
ptr = x.data_ptr()
size = x.numel() * x.element_size()

# ctypes를 통해 cudaMemPrefetchAsync 직접 호출 (실험용)
libcudart = ctypes.CDLL('libcudart.so')
stream = torch.cuda.current_stream().cuda_stream
device = torch.cuda.current_device()

err = libcudart.cudaMemPrefetchAsync(ctypes.c_void_p(ptr), ctypes.c_size_t(size), ctypes.c_int(device), ctypes.c_void_p(stream))
assert err == 0, f"Prefetch failed with error code {err}"

# 4. GPU에서 직접 연산
x_cuda = x.to('cuda', non_blocking=True)
print((x_cuda + 1).mean())