#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "cuda_bfs.cuh"
using namespace std;
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
// // Function to print all elements in a vector
// void print_vector(const std::vector<int>& vec) {
//     std::cout << "[";
//     for (size_t i = 0; i < vec.size(); ++i) {
//         std::cout << vec[i];
//         if (i != vec.size() - 1) {
//             std::cout << ", ";
//         }
//     }
//     std::cout << "]" << std::endl;
// }


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
    
    // cout << "rank: " << rank << endl;
    // cout << "printing edges and offsets.." << endl;
    // print_vector(edges);
    // print_vector(offsets);
    if (rank == 0) {
        cuda_init(m, n, edges, offsets);

        std::vector<int> bfs_result;
        cuda_bfs(n, 0, bfs_result);

        std::cout << "BFS Traversal Order: ";
        for (int v : bfs_result) {
            if (v != -1) std::cout << v + 1 << " ";
        }
        std::cout << std::endl;

        cuda_cleanup();
    }

    MPI_Finalize();
    return 0;
}
