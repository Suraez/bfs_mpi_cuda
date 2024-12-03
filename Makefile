all:
	nvcc -arch=sm_50 -c cuda_bfs.cu -o cuda_bfs.o
	mpicxx -o mpi_cuda_bfs main.cpp cuda_bfs.o -lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
	mpirun -np 4 ./mpi_cuda_bfs


clean:
	rm -rf *.o