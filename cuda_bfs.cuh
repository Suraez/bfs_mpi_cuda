#ifndef CUDA_BFS_H
#define CUDA_BFS_H

#include <vector>

// Function declarations
void cuda_bfs(int num_vertices, int start_vertex, std::vector<int> &output);
void cuda_init(int m, int n, const std::vector<int> &edges, const std::vector<int> &offsets);
void cuda_cleanup();

#endif // CUDA_BFS_H
