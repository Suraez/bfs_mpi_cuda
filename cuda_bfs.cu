#include <cuda_runtime.h>
#include "cuda_bfs.cuh"

#define BLOCK_SIZE 256

__global__ void bfs_kernel(int *edges, int *offsets, int *visited, int *frontier, int frontier_size, int *next_frontier, int *next_frontier_size, int num_vertices) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= frontier_size) return;

    int current_vertex = frontier[tid];
    for (int i = offsets[current_vertex]; i < offsets[current_vertex + 1]; i++) {
        int neighbor = edges[i];
        if (atomicExch(&visited[neighbor], 1) == 0) {
            int idx = atomicAdd(next_frontier_size, 1);
            next_frontier[idx] = neighbor;
        }
    }
}

void cuda_bfs(int *d_edges, int *d_offsets, int num_vertices, int start_vertex, int *visited, int *output, int *output_size) {
    int *frontier, *next_frontier, *d_output_size;
    cudaMalloc(&frontier, num_vertices * sizeof(int));
    cudaMalloc(&next_frontier, num_vertices * sizeof(int));
    cudaMalloc(&d_output_size, sizeof(int));

    int h_output_size = 0;
    cudaMemcpy(output, &start_vertex, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(output_size, &h_output_size, sizeof(int), cudaMemcpyHostToDevice);

    int frontier_size = 1;
    cudaMemcpy(frontier, &start_vertex, sizeof(int), cudaMemcpyHostToDevice);

    while (frontier_size > 0) {
        cudaMemset(d_output_size, 0, sizeof(int));
        bfs_kernel<<<(frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_edges, d_offsets, visited, frontier, frontier_size, next_frontier, d_output_size, num_vertices);
        cudaDeviceSynchronize();

        cudaMemcpy(&frontier_size, d_output_size, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(frontier, next_frontier, frontier_size * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaFree(frontier);
    cudaFree(next_frontier);
    cudaFree(d_output_size);
}
