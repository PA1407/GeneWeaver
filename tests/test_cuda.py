
import numpy as np
from numba import cuda

@cuda.jit
def add_arrays(a, b, result):
    index = cuda.grid(1)

    if index < result.size:
        result[index] = a[index] + b[index]


def main():
    print("CUDA available:", cuda.is_available())

    a = np.arange(10, dtype=np.float32)
    b = np.arange(10, dtype=np.float32)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_result = cuda.device_array_like(a)

    threads_per_block = 32
    blocks_per_grid = (a.size + threads_per_block - 1) // threads_per_block

    add_arrays[blocks_per_grid, threads_per_block](
        d_a, d_b, d_result
    )

    result = d_result.copy_to_host()

    print("Array A:", a)
    print("Array B:", b)
    print("GPU Result:", result)

    expected = a + b
    assert np.array_equal(result, expected)

    print("GPU TEST PASSED!")


if __name__ == "__main__":
    main()
