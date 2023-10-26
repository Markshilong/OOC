import time
import torch
import numpy as np

for i in range(10):
    # Set the size of the tensor (adjust according to your use case)
    samleTensor = torch.rand(10000,10000)
    torch.save(samleTensor, 'sampleTensor.pt')

    # Calculate the memory usage
    memory_usage_bytes = samleTensor.element_size() * samleTensor.numel()

    # Convert bytes to megabytes for a more readable output
    memory_usage_mb = memory_usage_bytes / (1024 ** 2)

    # Print the results
    print("Tensor Shape:", samleTensor.shape)
    print("Tensor Data Type:", samleTensor.dtype)
    print("Memory Usage:", memory_usage_bytes, "bytes or", memory_usage_mb, "MB")


    # Fetching tensor from NVMe to CPU memory
    start_time = time.time()
    samleTensor_loaded = torch.load('sampleTensor.pt')
    end_time = time.time()
    fetch_time = end_time - start_time

    print(f"Time to fetch tensor from NVMe to CPU: {fetch_time} seconds")

    # Pushing tensor from CPU memory to GPU memory
    start_time = time.time()
    samleTensor = samleTensor.to('cuda')
    end_time = time.time()
    push_time = end_time - start_time

    print(f"Time to push tensor from CPU to GPU: {push_time} seconds")
