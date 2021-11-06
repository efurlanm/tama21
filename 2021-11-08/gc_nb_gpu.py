def stencil(n) :
    import numpy as np, math
    from time import time
    from numba import cuda

    # parameters
    # n            = 2400    # nxn grid <-- parameter passing
    energy       = 1       # energy to be injected per iteration
    niters       = 250     # number of iterations
    # initialize the data arrays
    anew         = np.zeros((n + 2, n + 2), np.float64)
    aold         = np.zeros((n + 2, n + 2), np.float64)
    # initialize three heat sources
    sources      = np.empty((3, 2), np.int32)
    sources[:,:] = [ [n//2, n//2], [n//3, n//3], [n*4//5, n*8//9] ]

    # configure blocks & grids
    ## set the number of threads in a block
    threads_per_block = (16, 16)
    ## calculate the number of thread blocks in the grid
    blocks_per_grid_x = math.ceil(aold.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(aold.shape[1] / threads_per_block[1])
    blocks_per_grid   = (blocks_per_grid_x, blocks_per_grid_y)

    # computationally intensive core
    @cuda.jit
    def kernel(A, B, sources, energy):
        n = A.shape[0] - 1
        i, j = cuda.grid(2)
        if (i > 0 and j > 0) and (i < n and j < n) :
            A[i,j]=.5*B[i,j]+.125*(B[i-1,j]+B[i+1,j]+B[i,j-1]+B[i,j+1])
            if ((sources[0, 0] == i and sources[0, 1] == j) or
                (sources[1, 0] == i and sources[1, 1] == j) or
                (sources[2, 0] == i and sources[2, 1] == j)):
                A[i, j] += energy

    # main routine
    t0 = -time()    # time measure
    t1 = 0
    t2 = 0

    t_ = time()
    # copy the arrays to the device
    anew_global_mem    = cuda.to_device(anew)
    aold_global_mem    = cuda.to_device(aold)
    sources_global_mem = cuda.to_device(sources)
    t2 += time() - t_

    for _ in range(0, niters, 2) :
        t_ = time()
        kernel[blocks_per_grid, threads_per_block](
            anew_global_mem, aold_global_mem, sources_global_mem, energy)
        kernel[blocks_per_grid, threads_per_block](
            aold_global_mem, anew_global_mem, sources_global_mem, energy)
        t1 += time() - t_

    t_ = time()
    # copy the result back to the host
    aold = aold_global_mem.copy_to_host()
    t2 += time() - t_

    # system total heat
    heat = np.sum(aold[1:-1, 1:-1])

    t0 += time()

    # show the result
    print(f"Heat: {heat:.4f}", end=" | ")
    print(f"Time: {t0:.4f} s", end=" | ")
    print(f"Kernel: {t1:.4f} s", end=" | ")
    print(f"Memory: {t2:.4f} s")
    
    return aold
