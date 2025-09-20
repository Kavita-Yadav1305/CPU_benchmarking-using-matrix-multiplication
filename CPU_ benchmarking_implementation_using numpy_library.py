# TO_DO: Call benchmark_matrix_multiplication with different matrix sizes to run the benchmark
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
    

