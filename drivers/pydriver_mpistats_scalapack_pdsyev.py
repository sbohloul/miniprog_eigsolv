from mpi4py import MPI
import pb11_mpistats_scalapack_kernels as pb11msk
import numpy as np
import matplotlib.pyplot as plt
import argparse

seed = 42
np.random.seed(seed=seed)


def perform_analysis(args):

    # ==============
    # Initialize mpi
    # ==============
    mpicomm = MPI.COMM_WORLD
    mpirank = mpicomm.Get_rank()
    mpisize = mpicomm.Get_size()

    # ========================
    # kernel launch parameters
    # ========================
    dry_run = args.dry_run
    mpi_region = args.mpi_region
    m = args.m
    mb = args.mb
    nb = args.nb
    nprow = args.nprow
    npcol = args.npcol
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

    # =======
    # dry-run
    # =======
    if dry_run:
        if mpirank == 0:
            print("DRY-RUN ----------")
            print(f"mpi_region: {mpi_region} \n"
                  f"nprow: {nprow}, npcol: {npcol}\n"
                  f"mb: {mb}, nb: {nb}\n"
                  f"m: {m}"
                  )
        return

    # ============
    # Input arrays
    # ============
    if (mpirank == 0):
        a = np.random.rand(m, m).astype(np.float64)
        a = .5 * (a + a.T)
        a = a.flatten()
        eigvec = 0 * a
    else:
        a = np.array([0], dtype=np.float64)
        eigvec = np.array([0], dtype=np.float64)

    eigval = np.zeros((1, m), dtype=np.float64)

    # =============
    # Launch kernel
    # =============
    pb11msk.pb11_mpistats_scalapack_pdsyev(
        mpi_region, nprow, npcol, a, eigval, eigvec, m, mb, nb
    )


    if mpirank == 0:
        eigval_ref, eigvec_ref = np.linalg.eigh(a.reshape(m, m))
        err_eigval = np.linalg.norm(
            eigval.flatten() - eigval_ref.flatten(), np.inf)
        err_eigvec = np.linalg.norm(
            np.abs(eigvec.flatten()) - np.abs(eigvec_ref.T.flatten()), np.inf)

        print(f"err_eigval: {err_eigval}, err_eigvec: {err_eigvec}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        kernel:
        - pdsyev
        analysis:
        - Generates a random real symmetric matrix from given input parameters
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("mpi_region",
                        type=int,
                        help="Number to tag region from which statistics are collected. It is appended to report folder of aps tool")
    parser.add_argument("m",
                        type=int,
                        help="Square matrix dimension")
    parser.add_argument("mb",
                        type=int,
                        help="Size of rows in the block",
                        )
    parser.add_argument("nb",
                        type=int,
                        help="Size of columns in the block",
                        )
    parser.add_argument("nprow",
                        type=int,
                        help="Size of rows in process grid"
                        )
    parser.add_argument("npcol",
                        type=int,
                        help="Number of columns in process grid"
                        )
    parser.add_argument('--dry-run',
                        action='store_true',
                        help="Simulate the process without making actual changes"
                        )

    args = parser.parse_args()
    perform_analysis(args)
