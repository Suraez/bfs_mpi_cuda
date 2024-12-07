#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include "cuda_bfs.cuh"
#include <chrono>
#include <iomanip>
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

void bfs_sequential(int n, const std::vector<int> &edges, const std::vector<int> &offsets, double &duration) {
    std::vector<bool> visited(n, false);
    std::queue<int> q;
    std::vector<int> traversal;

    auto start_time = std::chrono::high_resolution_clock::now();

    q.push(0); // Start BFS from vertex 1 (index 0)
    visited[0] = true;

    while (!q.empty()) {
        int current = q.front();
        q.pop();
        traversal.push_back(current + 1); // Convert back to 1-based indexing

        for (int i = offsets[current]; i < offsets[current + 1]; ++i) {
            int neighbor = edges[i];
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Print BFS traversal
    std::cout << "Sequential BFS Traversal Path: ";
    for (int vertex : traversal) {
        std::cout << vertex << " ";
    }
    std::cout << std::endl;
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

void print_timing_comparison(double seq_timer, double par_timer) {
    std::cout << "=============================================" << std::endl;
    std::cout << "| BFS Timing Comparison (in milliseconds)   |" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(2); // Fixed-point notation with 2 decimal places
    std::cout << "| Sequential BFS: | " << std::setw(10) << seq_timer << " ms           |" << std::endl;
    std::cout << "| Parallel BFS:   | " << std::setw(10) << par_timer << " ms           |" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    if (seq_timer > par_timer) {
        std::cout << "| Parallel BFS is faster by " 
                  << std::setw(10) << seq_timer - par_timer << " ms!       |" << std::endl;
    } else if (seq_timer < par_timer) {
        std::cout << "| Sequential BFS is faster by " 
                  << std::setw(10) << par_timer - seq_timer << " ms!       |" << std::endl;
    } else {
        std::cout << "| Both BFS approaches took the same time!   |" << std::endl;
    }
    std::cout << "=============================================" << std::endl;
}




int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    if (argc < 2) {
        if (MPI::COMM_WORLD.Get_rank() == 0) {
            std::cerr << "Graph File name not given." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string graph_filename = argv[1];

    int world_size, rank;
    double seq_timer, par_timer;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n, m;
    std::vector<int> edges, offsets;

    if (rank == 0) {
        read_graph(graph_filename, n, m, edges, offsets);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        offsets.resize(n + 1);
        edges.resize(m);
    }
    MPI_Bcast(offsets.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(edges.data(), m, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        auto start_time = std::chrono::high_resolution_clock::now();
        cuda_init(m, n, edges, offsets);


        // Perform Sequential BFS

        std::vector<int> bfs_result;
        cuda_bfs(n, 0, bfs_result);

        auto end_time = std::chrono::high_resolution_clock::now();
        par_timer = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        bfs_sequential(n, edges, offsets, seq_timer);
        std::cout << "BFS Traversal Order: ";
        for (int v : bfs_result) {
            if (v != -1) std::cout << v + 1 << " ";
        }
        std::cout << std::endl;

        print_timing_comparison(seq_timer, par_timer);
        cuda_cleanup();
    }

    MPI_Finalize();
    return 0;
}
