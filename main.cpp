#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "cuda_bfs.cuh"

void read_graph(const std::string &filename, int &n, int &m, std::vector<int> &edges, std::vector<int> &offsets) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(1);
    }

    file >> n >> m;
    offsets.resize(n + 1, 0);
    edges.reserve(m);

    int u, v;
    std::vector<std::vector<int>> adjacency_list(n);
    while (file >> u >> v) {
        adjacency_list[u - 1].push_back(v - 1);
    }

    for (int i = 0; i < n; i++) {
        offsets[i + 1] = offsets[i] + adjacency_list[i].size();
        edges.insert(edges.end(), adjacency_list[i].begin(), adjacency_list[i].end());
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n, m;
    std::vector<int> edges, offsets;

    if (rank == 0) {
        read_graph("graph.txt", n, m, edges, offsets);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        offsets.resize(n + 1);
        edges.resize(m);
    }
    MPI_Bcast(offsets.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(edges.data(), m, MPI_INT, 0, MPI_COMM_WORLD);

    int *d_edges, *d_offsets, *d_visited, *d_output, d_output_size;
    cudaMalloc(&d_edges, m * sizeof(int));
    cudaMalloc(&d_offsets, (n + 1) * sizeof(int));
    cudaMalloc(&d_visited, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_edges, edges.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_visited, 0, n * sizeof(int));

    int start_vertex = 0;
    std::vector<int> local_bfs_result(n);
    cuda_bfs(d_edges, d_offsets, n, start_vertex, d_visited, d_output, &d_output_size);

    cudaMemcpy(local_bfs_result.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    if (rank == 0) {
        std::cout << "BFS Traversal Order: ";
        for (int v : local_bfs_result) {
            if (v != -1) std::cout << v + 1 << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_edges);
    cudaFree(d_offsets);
    cudaFree(d_visited);
    cudaFree(d_output);

    MPI_Finalize();
    return 0;
}
