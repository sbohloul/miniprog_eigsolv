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
    MB_MIN = 64
    NB_MIN = 64

    dry_run = args.dry_run
    outfile = args.output + ".csv"
    outplot = args.output + ".svg"
    niter = args.niter
    mb_max = args.mb
    nb_max = args.nb
    nprow = args.nprow
    npcol = args.npcol
    #
    m = nprow * mb_max
    n = npcol * nb_max
    nprocs = nprow * npcol

    # ========================
    # kernel launch parameters
    # ========================
    mb_list = [mb_max]
    x = mb_max
    while x > MB_MIN:
        x = int(x / 2)
        mb_list.insert(0, x)

    nb_list = [nb_max]
    x = nb_max
    while x > NB_MIN:
        x = int(x / 2)
        nb_list.insert(0, x)

    # =======
    # dry-run
    # =======
    if dry_run:
        if mpirank == 0:
            print("DRY-RUN ----------")
            print(f"niter: {niter} \n"
                  f"m: {m}, n: {n}\n"
                  f"nprow: {nprow}, npcol: {npcol}\n"
                  f"mb_max: {mb_max}, nb_max: {nb_max}\n"
                  f"mb: {mb_list}\n"
                  f"nb: {nb_list}\n"
                  )
        return

    # ===================
    # Verify process grid
    # ===================
    if (mpisize != nprocs):
        if mpirank == 0:
            raise ValueError(
                f"mpisize: {mpisize} != (npcol * nprow): {nprocs}")
        else:
            exit()

    # ============
    # Input arrays
    # ============
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

    # =============
    # Launch kernel
    # =============
    tkernel = []
    for mb, nb in zip(mb_list, nb_list):
        t_tmp = pb11tsk.pb11_time_scalapack_pdgemm(
            niter, nprow, npcol, a, b, c, m, n, mb, nb)
        tkernel.append(t_tmp)

    # Gathe all timings in rank 0
    tkernel = mpicomm.gather(tkernel, root=0)

    # ==============
    # Output results
    # ==============
    # write data to output
    if mpirank == 0:
        df = pd.DataFrame(tkernel, columns=[x for x in mb_list])
        df.index.name = 'rank'
        df.columns.name = 'blocksize'
        print(df)
        with open(outfile, 'w') as f:
            f.write(
                "# time vs num blocks per process\n"
                f"# nprocs: {nprocs}\n"
                f"# nprow: {nprow} , npcol: {npcol}\n"
                f"# mb_max: {mb_max}, nb_max: {nb_max}\n"
                f"# m: {m}, n: {n}\n"
            )
            df.to_csv(f, sep='\t', index=True,
                      header=True, float_format='%.6f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        kernel:
        - pdgemm
        analysis:
        - Fixed process grid p x q
        - Fixed global array size m x n
        - Measure timing for varying block size mb x nb where max_mb and max_nb are passed as input
        - mb = [64, 128, 256, ..., max_mb]
        - nb = [64, 128, 256, ..., max_nb]
        - m = max(mb) * p, n = max(nb) * q        
        """
    )

    parser.add_argument(
        "niter", type=int, help="Number of iterations used for averaging the timing")
    parser.add_argument("mb",
                        type=int,
                        help="Maximum number of rows in the block",
                        choices=[64, 128, 256, 512, 1024, 2048, 4096]
                        )
    parser.add_argument("nb",
                        type=int,
                        help="Maximum number of columns in the block",
                        choices=[64, 128, 256, 512, 1024, 2048, 4096]
                        )
    parser.add_argument("nprow",
                        type=int,
                        help="Number of rows in process grid"
                        )
    parser.add_argument("npcol",
                        type=int,
                        help="Number of columns in process grid"
                        )
    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output file for exporting the results as text, figure, and etc",
                        default="time_vs_blocksize"
                        )
    parser.add_argument('--dry-run',
                        action='store_true',
                        help="Simulate the process without making actual changes"
                        )

    args = parser.parse_args()
    perform_analysis(args)
