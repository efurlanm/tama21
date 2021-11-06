import numpy as np
from time import time
from numba import njit, set_num_threads, get_num_threads
from mpi4py import MPI   

# computationally intensive core
@njit('(float64[:,:],float64[:,:])', parallel=True, fastmath=True, nogil=True)
def kernel(anew, aold) :
    anew[1:-1,1:-1] = (aold[1:-1,1:-1]/2.0
        +(aold[2:,1:-1]+aold[:-2,1:-1]+aold[1:-1,2:]+aold[1:-1,:-2])/8.0)

# parameters
n            = 2400    # n x n grid
energy       = 1.0     # energy to be injected per iteration
niters       = 250     # number of iterations
# initialize three heat sources
nsources     = 3    # sources of energy
sources      = np.zeros((nsources, 2), np.int16)
sources[:,:] = [ [n//2, n//2], [n//3, n//3], [n*4//5, n*8//9] ]

# main routine
comm    = MPI.COMM_WORLD
mpisize = comm.size
mpirank = comm.rank
if not mpirank : t0 = -time()

# determine my coordinates (x,y)
pdims = MPI.Compute_dims(mpisize, 2)
px    = pdims[0]
py    = pdims[1]
rx    = mpirank % px
ry    = mpirank // px

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
    # wait
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

    # update grid
    kernel(anew, aold)

    # refresh heat sources
    anew[locsources[:locnsources, 0], locsources[:locnsources, 1]] += energy

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
    # wait
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

    # update grid
    kernel(aold, anew)

    # refresh heat sources
    aold[locsources[:locnsources, 0], locsources[:locnsources, 1]] += energy 

# get final heat in the system
bheat[0] = np.sum(aold[1:-1, 1:-1])
comm.Reduce(bheat, rheat)

if not mpirank :
    t0 += time()
    print(f"Heat: {rheat[0]:.4f}", end=" | ")
    print(f"Time: {t0:.4f}", end=" | ")
    print(f"MPISize: {mpisize}")
