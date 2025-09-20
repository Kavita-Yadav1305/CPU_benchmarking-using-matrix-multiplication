# CPU_benchmarking-using-matrix-multiplication


##### -> Implemented a CPU benchmarking tool by measuring execution time of matrix multiplications for increasing matrix sizes.

##### -> Built a naïve O(n³) Python implementation and optimized it using NumPy’s BLAS-backed dot function, achieving 100x+ speedup.

##### -> Automated benchmarking for multiple matrix sizes (50×50 to 1000×1000) and analyzed performance differences.

##### -> Demonstrated performance engineering, algorithm optimization, and efficient numerical computing practices.

## First we have implemnted without using numpy
```
import random
import time

def generate_matrix1(size):
    # TO_DO: Generate a matrix filled with random numbers
    return [[random.random() for _ in range(size)] for _ in range(size)]

def matrix_multiply1(matrix1, matrix2):
    # TO_DO: Implement matrix multiplication
    size = len(matrix1)
    result = [[0 for _ in range(size)] for _ in range(size)]
 
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
 
    return result

def benchmark_matrix_multiplication1(size):
    # TO_DO: Benchmark the matrix multiplication function

    # Generate two matrices of the given size
    # TO_DO: Call generate_matrix for matrix1 and matrix2

    # Get the start time
    # TO_DO: Use time.time() to get the start_time

    # Perform matrix multiplication
    # TO_DO: Call matrix_multiply with matrix1 and matrix2 and store the result

    # Get the end time
    # TO_DO: Use time.time() to get the end_time

    # Calculate and print the elapsed time
    # TO_DO: Subtract start_time from end_time to get elapsed_time
    # TO_DO: Print the elapsed time with a message
    matrix1 = generate_matrix1(size)
    matrix2 = generate_matrix1(size)
 
    start_time = time.time()
    result = matrix_multiply1(matrix1, matrix2)
    end_time = time.time()
 
    elapsed_time = end_time - start_time
    print(f"Elapsed time for {size}x{size} matrix multiplication: {elapsed_time} seconds")
# TO_DO: Call benchmark_matrix_multiplication with different matrix sizes to run the benchmark
if __name__ == "__main__":
    for size in [50, 100, 150, 500, 1000]:
        benchmark_matrix_multiplication1(size)
```
## Now we have implemented the same using numpy library
```
import time
import numpy as np

def generate_matrix(size):
    """Generate a random matrix using NumPy for efficiency."""
    return np.random.rand(size, size)

def matrix_multiply(matrix1, matrix2):
    """Multiply matrices using NumPy's optimized dot product."""
    return np.dot(matrix1, matrix2)

def benchmark_matrix_multiplication(size):
    """Benchmark matrix multiplication for a given size."""
    matrix1 = generate_matrix(size)
    matrix2 = generate_matrix(size)

    start_time = time.time()
    result = matrix_multiply(matrix1, matrix2)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time for {size}x{size} matrix multiplication: {elapsed_time:.6f} seconds")
    return elapsed_time


if __name__ == "__main__":
    for size in [50, 100, 150, 500, 1000]:
        benchmark_matrix_multiplication(size)
```

