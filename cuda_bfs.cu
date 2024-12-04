#include <cuda_runtime.h>
#include "cuda_bfs.cuh"
#include<stdio.h>
#define BLOCK_SIZE 256

// Device pointers to be shared across functions
int *d_edges, *d_offsets, *d_visited, *d_output;

// CUDA Initialization Function
void cuda_init(int m, int n, const std::vector<int> &edges, const std::vector<int> &offsets) {
    cudaMalloc(&d_edges, m * sizeof(int));
    cudaMalloc(&d_offsets, (n + 1) * sizeof(int));
    cudaMalloc(&d_visited, (n + 1) * sizeof(int)); // Extra slot for traversal counter
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_edges, edges.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
}

// CUDA Cleanup Function
void cuda_cleanup() {
    cudaFree(d_edges);
    cudaFree(d_offsets);
    cudaFree(d_visited);
    cudaFree(d_output);
}

// BFS Kernel
__global__ void bfs_kernel(int *edges, int *offsets, int *visited, int *frontier, int frontier_size, int *next_frontier, int *next_frontier_size, int num_vertices, int *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= frontier_size) return;

    int current_vertex = frontier[tid];
    for (int i = offsets[current_vertex]; i < offsets[current_vertex + 1]; i++) {
        int neighbor = edges[i];

        // Only add unvisited neighbors to the next frontier
        if (atomicExch(&visited[neighbor], 1) == 0) {
            int idx = atomicAdd(next_frontier_size, 1);
            next_frontier[idx] = neighbor;

            // Add to the output array
            int output_idx = atomicAdd(&visited[num_vertices], 1); // Use visited[num_vertices] as a counter
            output[output_idx] = neighbor;

            // Debugging print
            // printf("Adding vertex %d to output at index %d\n", neighbor + 1, output_idx);
        }
    }
}

// BFS Function
void cuda_bfs(int num_vertices, int start_vertex, std::vector<int> &output) {
    int *frontier, *next_frontier, *d_output_size;
    cudaMalloc(&frontier, num_vertices * sizeof(int));
    cudaMalloc(&next_frontier, num_vertices * sizeof(int));
    cudaMalloc(&d_output_size, sizeof(int));

    // Initialize visited and output arrays
    cudaMemset(d_visited, 0, (num_vertices + 1) * sizeof(int)); // Extra slot for traversal counter
    cudaMemset(d_output, -1, num_vertices * sizeof(int));
    cudaMemset(d_output_size, 0, sizeof(int));

    // Add the start vertex to the frontier and mark it visited
    int frontier_size = 1;
    cudaMemcpy(frontier, &start_vertex, sizeof(int), cudaMemcpyHostToDevice);

    // Host array for visited
    int *h_visited = new int[num_vertices + 1](); // +1 for traversal counter
    h_visited[start_vertex] = 1;                 // Mark the start vertex as visited
    h_visited[num_vertices] = 1;                 // Initialize traversal counter to 1
    cudaMemcpy(d_visited, h_visited, (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Host array for output
    int *h_output = new int[num_vertices];       // Host array for output
    std::fill(h_output, h_output + num_vertices, -1); // Initialize to -1
    h_output[0] = start_vertex;                  // Add the start vertex at the beginning
    cudaMemcpy(d_output, h_output, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

    // Perform BFS
    while (frontier_size > 0) {
        cudaMemset(d_output_size, 0, sizeof(int));

        bfs_kernel<<<(frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_edges, d_offsets, d_visited, frontier, frontier_size, next_frontier, d_output_size, num_vertices, d_output);
        cudaDeviceSynchronize();

        cudaMemcpy(&frontier_size, d_output_size, sizeof(int), cudaMemcpyDeviceToHost);

        if (frontier_size > 0) {
            cudaMemcpy(frontier, next_frontier, frontier_size * sizeof(int), cudaMemcpyDeviceToDevice);
        }
    }

    // Copy results back to host
    output.resize(num_vertices);
    cudaMemcpy(output.data(), d_output, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup host memory
    delete[] h_visited;
    delete[] h_output;

    // Free allocated device memory
    cudaFree(frontier);
    cudaFree(next_frontier);
    cudaFree(d_output_size);
}