from mpi4py import MPI
import pb11_time_scalapack_kernels as pb11tsk
import numpy as np
from scipy.linalg import eigh, eig
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import h5py
import os


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
    fname = args.inputfile
    #
    nprocs = nprow * npcol
    m = None

    # =============================
    # Check whether inputfile exist
    # =============================
    if os.path.exists(fname):
        if mpirank == 0:
            print(f"{fname} exists")
    else:
        if mpirank == 0:
            raise FileNotFoundError(f"The file '{fname}' does not exist")
        else:
            exit()

    # and can be read by h5py
    infile_is_readable = None
    if mpirank == 0:
        try:
            with h5py.File(fname, 'r') as h5file:
                infile_is_readable = True
                print(f"{fname} can be read")
        except Exception as e:
            pass
            # print(f"Error in reading {fname}: {e}")

    infile_is_readable = mpicomm.bcast(infile_is_readable, root=0)
    if infile_is_readable == False:
        if mpirank == 0:
            raise f"Error in reading {fname}: {e}"
        else:
            exit()

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
                  f"nprow: {nprow}, npcol: {npcol}\n"
                  f"mb_max: {mb_max}, nb_max: {nb_max}\n"
                  f"mb: {mb_list}\n"
                  f"nb: {nb_list}\n"
                  f"inputfile: {fname}"
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
        f = h5py.File(fname, 'r')
        h = f['hamiltonian']['total_gamma'][:]
        o = f['hamiltonian']['overlap_gamma'][:]
        m = h.shape[0]
        print(f"m: {m}")

        Linv = np.linalg.inv(np.linalg.cholesky(o))
        a = np.dot(Linv, h)
        a = np.dot(a, Linv.T)
        # a = .5 * (a + a.T)
        a = a.flatten()
        eigvec = 0 * a
    else:
        # init dummy arrays
        a = np.array([0], dtype=np.float64)
        eigvec = np.array([0], dtype=np.float64)

    m = mpicomm.bcast(m, root=0)
    eigval = np.zeros((1, m), dtype=np.float64)

    # =============
    # Launch kernel
    # =============
    tkernel = []
    for mb, nb in zip(mb_list, nb_list):
        t_tmp = pb11tsk.pb11_time_scalapack_pdsyev(
            niter, nprow, npcol, a, eigval, eigvec, m, mb, nb)
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
                "# time vs block size\n"
                f"# nprocs: {nprocs}\n"
                f"# nprow: {nprow} , npcol: {npcol}\n"
                f"# mb_max: {mb_max}, nb_max: {nb_max}\n"
                f"# m: {m}\n"
            )
            df.to_csv(f, sep='\t', index=True,
                      header=True, float_format='%.6f')

    # Plot time vs block size
    if mpirank == 0:
        xlabel = "Block size"
        ylabel = "Time (s)"
        title = f"pdsyev \n mb: {mb}, nb: {nb} \n m: {m}"

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(title)
        for iproc in range(nprocs):
            axs[0].plot(
                mb_list, tkernel[iproc],
                label="rank " + str(iproc),
                marker='o'
            )
            axs[1].loglog(
                mb_list, tkernel[iproc],
                label="rank " + str(iproc),
                marker='o'
            )
        for ax in axs:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid()
            ax.legend()
        plt.tight_layout()
        plt.savefig(outplot)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        kernel:
        - pdsyev
        analysis:
        - It reads corresponding Hamiltonian and Overlap matrix from hdf5 input file
        - Measure timing for varying block size mb x nb where max_mb and max_nb are passed as input
        - mb = [64, 128, 256, ..., max_mb]
        - nb = [64, 128, 256, ..., max_nb]
        - Process grid p x q is passes as inputs
        """,
        formatter_class=argparse.RawTextHelpFormatter
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
    parser.add_argument("inputfile",
                        type=str,
                        help="Path to hdf5 file for Hamilotnian and Overlap matrix"
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
