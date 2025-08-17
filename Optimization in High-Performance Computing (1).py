import numpy as np
import time

# Define matrix size
N = 1000  # Large enough to observe performance differences

# Generate a large random NxN matrix
matrix = np.random.rand(N, N)

# -----------------------------------------------------------------------------
# 🟥 1️⃣ Inefficient Memory Access (Column-Major Order) 🟥
# -----------------------------------------------------------------------------
def inefficient_memory_access(matrix):
    """
    Traverse the matrix in a column-major order (poor cache locality).
    This leads to frequent cache misses, slowing down performance.
    """
    sum_value = 0
    for col in range(matrix.shape[1]):  # Traverse by columns first
        for row in range(matrix.shape[0]):  # Then by rows
            sum_value += matrix[row, col]  # Accessing elements column-wise
    return sum_value


# -----------------------------------------------------------------------------
# 🟩 2️⃣ Optimized Memory Access (Row-Major Order) 🟩
# -----------------------------------------------------------------------------
def optimized_memory_access(matrix):
    """
    Traverse the matrix in a row-major order (better cache locality).
    This ensures adjacent elements are accessed together, improving cache performance.
    """
    sum_value = 0
    for row in range(matrix.shape[0]):  # Traverse by rows first
        for col in range(matrix.shape[1]):  # Then by columns
            sum_value += matrix[row, col]  # Accessing elements row-wise
    return sum_value


# -----------------------------------------------------------------------------
# 🟦 3️⃣ Loop Unrolling Optimization 🟦
# -----------------------------------------------------------------------------
def loop_unrolling_optimization(matrix):
    """
    Loop unrolling reduces loop overhead by processing multiple elements per iteration.
    This exploits instruction-level parallelism and improves CPU efficiency.
    """
    sum_value = 0
    rows, cols = matrix.shape
    
    for row in range(rows):
        col = 0
        while col < cols - 3:  # Unroll by a factor of 4
            sum_value += matrix[row, col] + matrix[row, col + 1] + matrix[row, col + 2] + matrix[row, col + 3]
            col += 4
        
        # Handle remaining elements
        while col < cols:
            sum_value += matrix[row, col]
            col += 1
            
    return sum_value


# -----------------------------------------------------------------------------
# 🟨 4️⃣ Blocking (Tiling) Optimization 🟨
# -----------------------------------------------------------------------------
def blocking_optimization(matrix, block_size=32):
    """
    Blocking (tiling) improves cache efficiency by processing small blocks of data.
    This ensures each block fits in the CPU cache, reducing memory access latency.
    """
    sum_value = 0
    rows, cols = matrix.shape

    for row_block in range(0, rows, block_size):
        for col_block in range(0, cols, block_size):
            for row in range(row_block, min(row_block + block_size, rows)):
                for col in range(col_block, min(col_block + block_size, cols)):
                    sum_value += matrix[row, col]
                    
    return sum_value


# -----------------------------------------------------------------------------
# 🟣 5️⃣ Vectorization with NumPy (Highly Optimized) 🟣
# -----------------------------------------------------------------------------
def vectorized_numpy_sum(matrix):
    """
    Uses NumPy's built-in vectorized operations to sum elements efficiently.
    NumPy internally optimizes memory access and leverages SIMD.
    """
    return np.sum(matrix)  # Leverages hardware acceleration


# -----------------------------------------------------------------------------
# 🏁 Performance Benchmarking 🏁
# -----------------------------------------------------------------------------
def benchmark_functions():
    """
    Runs and times all the optimization functions to compare their performance.
    """
    functions = {
        "Inefficient Memory Access (Column-Major)": inefficient_memory_access,
        "Optimized Memory Access (Row-Major)": optimized_memory_access,
        "Loop Unrolling Optimization": loop_unrolling_optimization,
        "Blocking (Tiling) Optimization": blocking_optimization,
        "Vectorized NumPy Sum": vectorized_numpy_sum
    }

    results = {}

    for name, func in functions.items():
        start_time = time.time()
        result = func(matrix)  # Execute the function
        end_time = time.time()

        execution_time = end_time - start_time
        results[name] = execution_time
        print(f"{name}: {execution_time:.5f} seconds (Sum: {result:.2f})")

    # Performance Comparison
    baseline = results["Inefficient Memory Access (Column-Major)"]
    for name, exec_time in results.items():
        if name != "Inefficient Memory Access (Column-Major)":
            speedup = baseline / exec_time
            print(f"✅ {name} is {speedup:.2f}x faster than Column-Major Order.")

# Run benchmark
benchmark_functions()


