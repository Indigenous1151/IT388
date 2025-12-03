How to run the code:
 Change directory to code
 
OMP:
 Compile with: g++ -std=c++17 -g -o omp ompShortestPath.cpp -fopenmp -O3
 <<< The -O3 flag is an optimization flag to improve performance >>>
 <<< The -std=c++17 flag specifies the compiler to use C++17 >>>

  Execute with ./omp <# Threads> ../graph/<input filename> <[1|0] display progress in console>

MPI:
 Compile with: mpic++ -std=c++17 -g -o mpi mpiShortestPath.cpp -O3
 <<< The -O3 flag is an optimization flag to improve performance >>>
 <<< The -std=c++17 flag specifies the compiler to use C++17 >>>
 
 Execute with mpiexec -np <# Threads> ./mpi ../graph/<input filename> <[1|0] display progress in console>
