import numpy as np, math
from time import time
from mpi4py import MPI
from numba import cuda, njit, prange, config

# parameters
n            = 4800    # n x n grid
energy       = 1.0     # energy to be injected per iteration
niters       = 500     # number of iterations
# initialize three heat sources
nsources     = 3       # number of sources of energy
sources      = np.zeros((nsources, 2), np.int16)
sources[:,:] = [ [n//2, n//2], [n//3, n//3], [n*4//5, n*8//9] ]
# initialize the data arrays
anew         = np.zeros((n + 2, n + 2), np.float64)
aold         = np.zeros((n + 2, n + 2), np.float64)

# configure blocks & grids
## set the number of threads in a block
threads_per_block = (16, 16)    # based on trial and error
## calculate the number of thread blocks in the grid
blocks_per_grid_x = math.ceil(aold.shape[0] / threads_per_block[0])
blocks_per_grid_y = math.ceil(aold.shape[1] / threads_per_block[1])
blocks_per_grid   = (blocks_per_grid_x, blocks_per_grid_y)

# computationally intensive core
@cuda.jit
def kernel(A, B):
    n = A.shape[0] - 1
    i, j = cuda.grid(2)
    if (i > 0 and j > 0) and (i < n and j < n) :
        A[i,j]=B[i,j]*.5+(B[i-1,j]+B[i+1,j]+B[i,j-1]+B[i,j+1])*.125

# start of main routine

#---mpi4py---
comm  = MPI.COMM_WORLD            # MPI default communicator
size  = comm.Get_size()           # MPI size
rank  = comm.Get_rank()           # MPI rank
name  = MPI.Get_processor_name()  # core hostname (eg sdumont3170)

#Only 2 processes per node are selected via Slurm. Within a node, color 
#rank 0 corresponds to the first process of this node, and color rank 1 
#corresponds to the second process of this node, and the other nodes are 
#similar. Example:
#  node      rank  color rank
#----------- ----  ----------
#sdumont3170   0        0
#sdumont3170   1        1
#sdumont3171   2        0
#sdumont3171   3        1
#sdumont3172   4        0
#sdumont3172   5        1
#sdumont3173   6        0
#sdumont3173   7        1
for i, c in enumerate(name) :     # find first digit in hostname
    if c.isdigit() :
        break
mcol  = int(name[i:])             # extract number from hostname
scomm = comm.Split(color = mcol)  # new communicator for the node
crank = scomm.Get_rank()          # get the node color rank

#---numba.cuda---
#In this implementation, Slurm is configured to run only 2 processes on 
#each node. For each of these processes (cores), a single GPU is 
#associated. Thus, within a node, color rank 0 is associated with GPU 0, 
#and color rank 1 is associated with GPU 1.
cuda.select_device(crank)         # 'color rank' 0 = 'gpu id' 0, etc.
cid = cuda.current_context().device.id

# time measurement for rank 0
if not rank :
    tt = -time()    # rank 0 time
    tk = 0          # accumulate kernel time
    tc = 0          # accumulate communication time

# determine my coordinates (x,y)
pdims = MPI.Compute_dims(size, 2)
px    = pdims[0]
py    = pdims[1]
rx    = rank % px
ry    = rank // px

# determine my four neighbors
north = (ry - 1) * px + rx
if (ry - 1) < 0 :
    north = MPI.PROC_NULL
south = (ry + 1) * px + rx
if (ry + 1) >= py :
    south = MPI.PROC_NULL
west = ry * px + rx - 1
if (rx - 1) < 0 :
    west = MPI.PROC_NULL
east = ry * px + rx + 1
if (rx + 1) >= px :
    east = MPI.PROC_NULL

# decompose the domain
bx   = n // px          # block size in x
by   = n // py          # block size in y
offx = rx * bx + 1      # offset in x
offy = ry * by + 1      # offset in y

# sources in my area, local to my rank
locnsources = 0
locsources  = np.empty((nsources, 2), np.int16)

# determine which sources are in my patch
for i in range(nsources) :
    locx = sources[i, 0] - offx
    locy = sources[i, 1] - offy
    if(locx >= 0 and locx <= bx and locy >= 0 and locy <= by) :
        locsources[locnsources, 0] = locx
        locsources[locnsources, 1] = locy
        locnsources += 1

# working arrays with 1-wide halo zones
anew = np.zeros((bx+2, by+2), np.float64)
aold = np.zeros((bx+2, by+2), np.float64)

# system total heat
rheat = np.zeros(1, np.float64)
bheat = np.zeros(1, np.float64)

# copy the first arrays to the device
if not rank : tc -= time()
anew_global_mem    = cuda.to_device(anew)
aold_global_mem    = cuda.to_device(aold)
if not rank : tc += time()
   
# main loop
for _ in range(0, niters, 2) :

    # exchange data with neighbors
    if north != MPI.PROC_NULL :
        r1=comm.irecv(source=north, tag=1)
        s1=comm.isend(aold[1, 1:bx+1], dest=north, tag=1)
    if south != MPI.PROC_NULL :
        r2=comm.irecv(source=south, tag=1)
        s2=comm.isend(aold[bx, 1:bx+1], dest=south, tag=1)
    if east != MPI.PROC_NULL :
        r3 = comm.irecv(source=east, tag=1)
        s3 = comm.isend(aold[1:bx+1, bx], dest=east, tag=1)
    if west != MPI.PROC_NULL :
        r4 = comm.irecv(source=west, tag=1)
        s4 = comm.isend(aold[1:bx+1, 1], dest=west, tag=1)
    # wait for the end of communication
    if north != MPI.PROC_NULL :
        s1.wait()
        aold[0, 1:bx+1] = r1.wait()
    if south != MPI.PROC_NULL :
        s2.wait()
        aold[bx+1, 1:bx+1] = r2.wait()
    if east != MPI.PROC_NULL :
        s3.wait()
        aold[1:bx+1, bx+1] = r3.wait()
    if west != MPI.PROC_NULL :
        s4.wait
        aold[1:bx+1, 0] = r4.wait()

    # copy the received array to the device
    if not rank : tc -= time()
    aold_global_mem = cuda.to_device(aold)
    if not rank : tc += time()
        
    # update grid
    if not rank : tk -= time()
    kernel[blocks_per_grid, threads_per_block](
        anew_global_mem, aold_global_mem)
    if not rank : tk += time()
        
    # copy the result back to the host
    if not rank : tc -= time()
    anew = anew_global_mem.copy_to_host()
    if not rank : tc += time()
        
    # refresh heat sources
    for i in range(locnsources) :
        anew[locsources[i, 0]-1, locsources[i, 1]-1] += energy

    # exchange data with neighbors
    if north != MPI.PROC_NULL :
        r1=comm.irecv(source=north, tag=1)
        s1=comm.isend(anew[1, 1:bx+1], dest=north, tag=1)
    if south != MPI.PROC_NULL :
        r2=comm.irecv(source=south, tag=1)
        s2=comm.isend(anew[bx, 1:bx+1], dest=south, tag=1)
    if east != MPI.PROC_NULL :
        r3 = comm.irecv(source=east, tag=1)
        s3 = comm.isend(anew[1:bx+1, bx], dest=east, tag=1)
    if west != MPI.PROC_NULL :
        r4 = comm.irecv(source=west, tag=1)
        s4 = comm.isend(anew[1:bx+1, 1], dest=west, tag=1)
    # wait for the end of communication
    if north != MPI.PROC_NULL :
        s1.wait()
        anew[0, 1:bx+1] = r1.wait()
    if south != MPI.PROC_NULL :
        s2.wait()
        anew[bx+1, 1:bx+1] = r2.wait()
    if east != MPI.PROC_NULL :
        s3.wait()
        anew[1:bx+1, bx+1] = r3.wait()
    if west != MPI.PROC_NULL :
        s4.wait
        anew[1:bx+1, 0] = r4.wait()

    # copy the received array to the device
    if not rank : tc -= time()
    anew_global_mem = cuda.to_device(anew)
    if not rank : tc += time()

    # update grid
    if not rank : tk -= time()
    kernel[blocks_per_grid, threads_per_block](
        aold_global_mem, anew_global_mem)
    if not rank : tk += time()
        
    # copy the result back to the host
    if not rank : tc -= time()
    aold = aold_global_mem.copy_to_host()
    if not rank : tc += time()
        
    # refresh heat sources
    for i in range(locnsources) :
        aold[locsources[i, 0]-1, locsources[i, 1]-1] += energy 

# end for

# get final heat in the system
bheat[0] = np.sum(aold[1:-1, 1:-1])
comm.Reduce(bheat, rheat)

# show the result
print(f"3. {name:11s}   {rank:02d}    {crank:02d}   {cid:02d}")
if not rank :
    tt += time()
    print( "1. hostname    rank crank  cid")
    print( "2. ----------- ---- ----- ----")
    print( "4. ---------------------------")
    print(f"5. Heat:{rheat[0]:.4f}", end=", ")
    print(f"TT:{tt:.4f}", end=", ")
    print(f"KT:{tk:.4f}", end=", ")
    print(f"CT:{tc:.4f}", end=", ")
    print(f"MPI:{size}", end=", ")
    print(f"dim:{n}", end=", ")
    print(f"ite:{niters}")
