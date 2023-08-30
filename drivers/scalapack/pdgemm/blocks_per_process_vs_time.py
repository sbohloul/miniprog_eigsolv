from mpi4py import MPI
import pb11_time_scalapack_kernels as pb11tsk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def perform_analysis(args):

    # ==============
    # Initialize mpi
    # ==============
    mpicomm = MPI.COMM_WORLD
    mpirank = mpicomm.Get_rank()
    mpisize = mpicomm.Get_size()

    # ==================================
    # blacs and kernel launch parameters
    # ==================================

    outfile = args.output
    outplot = args.plotout + ".svg"
    niter = args.niter
    mb = args.mb
    nb = args.nb
    nprow = args.nprow
    npcol = args.npcol
    max_num_block = args.max_num_blocks
    #
    nprocs = nprow * npcol

    # ===================
    # Verify process grid
    # ===================
    if (mpisize != nprocs):
        if mpirank == 0:
            raise ValueError(
                f"mpisize: {mpisize} != (npcol * nprow): {nprocs}")
        else:
            exit()

    # =============
    # Launch kernel
    # =============
    tkernel = []
    num_blocks_per_proc = []
    for nblk in range(1, max_num_block):

        # global arrays sizes
        num_mblocks = nblk
        num_nblocks = nblk
        num_blocks_per_proc.append(num_mblocks * num_nblocks)
        m = (nprow * mb) * num_mblocks
        n = (npcol * nb) * num_nblocks

        # init arrays
        if (mpirank == 0):
            # rank 0 holds the global arrays
            # They are considered as column-major arrays in c++ kernels
            #
            # a = [[0 0 0 ...]
            #      [1 1 1 ...]
            #      [...   ...]]
            #
            # b = [[0 1 2 ...]
            #      [0 1 2 ...]
            #      [...   ...]]
            #
            a = np.arange(m, dtype=np.float64)
            a = np.tile(a, (m, 1)).flatten()
            b = np.arange(m, dtype=np.float64).reshape(m, 1)
            b = np.tile(b, (1, m)).flatten()
            c = 0 * a
        else:
            # init dummy arrays
            a = np.array([0], dtype=np.float64)
            b = np.array([0], dtype=np.float64)
            c = np.array([0], dtype=np.float64)

        # time the kernel for each (m, n)
        t_tmp = pb11tsk.pb11_time_scalapack_pdgemm(
            niter, nprow, npcol, a, b, c, m, n, mb, nb)
        tkernel.append(t_tmp)

    # Gather all timings in rank 0
    tkernel = mpicomm.gather(tkernel, root=0)

    # ==============
    # Output results
    # ==============
    # write data to output
    if mpirank == 0:
        df = pd.DataFrame(tkernel, columns=[x for x in num_blocks_per_proc])
        df.index.name = 'rank'
        df.columns.name = 'numblocks'
        print(df)
        with open(outfile, 'w') as f:
            f.write(
                "# time vs num blocks per process\n"
                f"# nprocs: {nprocs}\n"
                f"# nprow: {nprow} , npcol: {npcol}\n"
                f"# mb: {mb}, nb: {nb}\n"
                f"# max_num_mblocks: {num_mblocks}, # max_num_nblocks: {num_nblocks}\n"
                f"# max_m: {m}, max_n: {n}\n"
            )
            df.to_csv(f, sep='\t', index=True,
                      header=True, float_format='%.6f')

    if mpirank == 0:
        fig, ax = plt.subplots()
        for iproc in range(nprocs):
            ax.plot(num_blocks_per_proc,
                    tkernel[iproc], label="rank " + str(iproc), marker='o')
        plt.legend()
        plt.savefig(outplot)
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        kernel: pdgemm - test number of blocks per process vs time for fixed block 
        size and process grid.
        """)

    parser.add_argument(
        "niter", type=int, help="Number of iterations used for averaging the timing")
    parser.add_argument("mb",
                        type=int,
                        help="Number of rows in the block"
                        )
    parser.add_argument("nb",
                        type=int,
                        help="Number of columns in the block"
                        )
    parser.add_argument("nprow",
                        type=int,
                        help="Number of rows in process grid"
                        )
    parser.add_argument("npcol",
                        type=int,
                        help="Number of columns in process grid"
                        )
    parser.add_argument("max_num_blocks",
                        type=int,
                        help="Maximum number of blocks per process"
                        )

    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output file for writing the results",
                        default="blocks_per_process_vs_time.out"
                        )
    parser.add_argument("-po", "--plotout",
                        type=str,
                        help="Create plot from data and save in PLOT",
                        default="blocks_per_process_vs_time"
                        )

    args = parser.parse_args()
    perform_analysis(args)
