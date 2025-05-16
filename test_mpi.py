# Broadcast communication
from mpi4py import MPI

# Initialize MPI communicator
comm = MPI.COMM_WORLD
# Get rank of current process
rank = comm.Get_rank()

# Define data to be broadcasted by rank 0
if rank == 0:
    data = {'key1': [7, 2.72, 2+3j], 'key2': ('abc', 'xyz')}
else:
    data = None

# Broadcast data from rank 0 to all other processes
data = comm.bcast(data, root=0)

# Print received data
print("Process", rank, "received data:", data)