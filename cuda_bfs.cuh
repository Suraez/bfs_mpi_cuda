#ifndef CUDA_BFS_H
#define CUDA_BFS_H

void cuda_bfs(int *d_edges, int *d_offsets, int num_vertices, int start_vertex, int *visited, int *output, int *output_size);

#endif // CUDA_BFS_H
