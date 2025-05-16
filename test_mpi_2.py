from mpi4py import MPI
import dedalus.public as d3
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Aquí tot el teu codi d'inicialització (bases, camps, problema)...

if rank == 0:
    print("Iniciant càlcul en", comm.Get_size(), "processos")

solver = problem.build_solver()

# Executa el càlcul
solver.solve_dense(solver.subproblems[0])
evals = np.sort(solver.eigenvalues)

if rank == 0:
    print("Eigenvalues:", evals.real)
