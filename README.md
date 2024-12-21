### CS 698 BFS Graph Traversal with the MPI and CUDA for Directed Graph

Name: Suraj Kumar Ojha <br>
NJIT ID: 31669171 <br>
Email: so299@njit.edu <br>


<b>1. Compile the code</b>

`make`

<b>2. Run the code (before running the code make sure you have hosts.txt file, sample file is provided with repository) </b>

Or, you can run the code on a single machine by eliminating the `--hostfile` flag

i. Running on multiple machines with hosts.txt file

`mpirun -np 4 --hostfile hosts.txt mpi_cuda_bfs graph5.txt`

ii. Running on the single machine

`mpirun -np 4 mpi_cuda_bfs graph5.txt`


<b>Notes: </b> <br>
i. here graph5.txt is the graph file name, you can pass any graph or graphs that comes with the repository like graph3.txt graph.txt

    e.g. `mpirun -np 4 --hostfile hosts.txt mpi_cuda_bfs graph.txt`


ii. For changing the number of processes, you can change the `-np` flag e.g.

`mpirun -np 2 mpi_cuda_bfs graph5.txt`
