# CUDA Memory Optimization – Experiment 6

## 1. Aim
To study and compare memory access optimization techniques in CUDA by implementing matrix transpose and matrix multiplication using GPU parallel processing.

## 2. Objectives
- Understand coalesced and uncoalesced memory access.
- Implement matrix transpose using CUDA.
- Compare naive and optimized memory access patterns.
- Measure execution time and memory bandwidth.
- Analyze GPU performance using CUDA.

## 3. Technologies Used
- **Programming Language:** CUDA C/C++
- **Platform:** Google Colab
- **GPU:** NVIDIA Tesla T4
- **Compiler:** NVCC
- **CUDA Version:** 13.0

## 4. Experiment Details

### Part A: Matrix Transpose
Two approaches are implemented:

1. **Uncoalesced Memory Access:** Threads access global memory in a non-contiguous pattern.
2. **Coalesced Memory Access:** Threads access adjacent memory locations to improve memory efficiency. Shared memory is used to optimize the transpose operation.

### Part B: Naive Matrix Multiplication
A basic CUDA matrix multiplication kernel is implemented to perform multiplication in parallel on the GPU.

## 5. Execution Results

The following results were obtained during execution on an NVIDIA Tesla T4 GPU.

| Operation | Execution Time |
|---|---:|
| Uncoalesced Transpose | 2.235 ms |
| Coalesced Transpose | 0.758 ms |
| Naive Matrix Multiplication | 8.971 ms |

**Transpose Speedup:** 2.95x

**Memory Bandwidth:**
- Uncoalesced: 60.1 GB/s
- Coalesced: 177.0 GB/s

All verification tests passed in this run.

## 6. Project Structure

```text
CUDA_Memory_Optimization/
│
├── exp6.cu
├── README.md
└── results/
    ├── gpu_configuration.png
    ├── cuda_compilation.png
    ├── execution_output.png
    └── github_repository.png
```

## 7. How to Execute

Run the following commands in an environment with an NVIDIA GPU and CUDA Toolkit installed:

```bash
nvcc exp6.cu -o exp6
./exp6
```

Google Colab with a supported NVIDIA GPU can also be used.

## 8. Conclusion

This experiment demonstrates that coalesced memory access can significantly improve CUDA performance compared with uncoalesced memory access. The matrix transpose implementation achieved a 2.95x speedup in the recorded run. It also demonstrates how CUDA enables parallel matrix operations using GPU processing.

## 9. Author

**Alsaba J Awati**

GitHub: [Alsaa-00](https://github.com/Alsaa-00)