from mpi4py import MPI
import pb11_mpistats_scalapack_kernels as pb11msk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import h5py
import os
import json


def perform_analysis(args):

    # ==============
    # Initialize mpi
    # ==============
    MPI.Pcontrol(0)
    mpicomm = MPI.COMM_WORLD
    mpirank = mpicomm.Get_rank()
    mpisize = mpicomm.Get_size()

    # ==================================
    # blacs and kernel launch parameters
    # ==================================
    MB_MIN = 8
    NB_MIN = 8

    dry_run = args.dry_run
    outfile = args.output + ".csv"
    outplot = args.output + ".svg"
    mb_max = args.mb
    nb_max = args.nb
    fname = args.inputfile
    #
    nprocs = mpisize
    m = None

    # =====================
    # Get factors of nprocs
    # =====================
    nprocs_factors = [[1, nprocs]]
    for i in range(2, int(np.sqrt(nprocs))+1):
        if nprocs % i == 0:
            r = nprocs // i
            nprocs_factors.append([i, r])
    # possible process grid
    proc_grid = nprocs_factors
    for p, q in nprocs_factors[::-1]:
        proc_grid.append([q, p])

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
            print(f"mb_max: {mb_max}, nb_max: {nb_max}\n"
                  f"mb: {mb_list}\n"
                  f"nb: {nb_list}\n"
                  f"inputfile: {fname}\n"
                  f"proc_grid: {proc_grid}"
                  )
        return

    # =============================
    # Generate tags for mpi regions
    # =============================
    mpi_region_list = {}
    for mb ,nb in zip(mb_list, nb_list):
        for p, q in proc_grid:
            tag = mb * 1000 + nb * 100 + p * 10 + q
            mpi_region_list[(mb, nb, p, q)] = tag
    
    if mpirank == 0:
        for k, v in mpi_region_list.items():
            print(f"launch config: {k}, mpi_regin: {v}")
        
        region_to_blacs_conf = {value: key for key, value in mpi_region_list.items()}
        fname_mpi_region = "mpi_regions.txt"        
        with open(fname_mpi_region, "w") as f:
            f.write(json.dumps(region_to_blacs_conf, indent=4))

    # ============
    # Input arrays
    # ============
    if (mpirank == 0):
        print(f"Preparing input arrays ...")
        f = h5py.File(fname, 'r')
        h = f['hamiltonian']['total_gamma'][:]
        o = f['hamiltonian']['overlap_gamma'][:]
        m = h.shape[0]
        print(f"m: {m}")

        # convert generalized eig problem to standard
        # A x = l B x -> L^-1 A L^-1^T x = l x
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
    # [ [(mb, nb), [[rank0], [rank1], ...]], ... ]
    # [rank0] = [t0, t1, t2, ..., tn], n = len(proc_grid)
    data_all = []
    for mb, nb in zip(mb_list, nb_list):
        tkernel = []
        for p, q in proc_grid:
            if mpirank == 0:
                print(f"Running kernel for mb: {mb}, nb: {nb}, p: {p}, q: {q}")
            nprow = p
            npcol = q
            mpi_region = mpi_region_list[(mb, nb, p, q)]
            pb11msk.pb11_mpistats_scalapack_pdsyev(
                mpi_region, nprow, npcol, a, eigval, eigvec, m, mb, nb)
            MPI.Pcontrol(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        kernel:
        - pdsyev
        analysis:
        - It reads corresponding Hamiltonian and Overlap matrix from hdf5 input file
        - Measure mpistats for all possible p x q grids for a given nprocs
        - mb = [64, 128, 256, ..., max_mb]
        - nb = [64, 128, 256, ..., max_nb]
        - It automatically tag the mpi region of interest based on (pxq) and (mbxnb)
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("mb",
                        type=int,
                        help="Maximum number of rows in the block",
                        choices=[8, 16, 32, 64, 128,
                                 256, 512, 1024, 2048, 4096]
                        )
    parser.add_argument("nb",
                        type=int,
                        help="Maximum number of columns in the block",
                        choices=[8, 16, 32, 64, 128,
                                 256, 512, 1024, 2048, 4096]
                        )
    parser.add_argument("inputfile",
                        type=str,
                        help="Path to hdf5 file for Hamilotnian and Overlap matrix"
                        )

    parser.add_argument("-o", "--output",
                        type=str,
                        help="Output file for exporting the results as text, figure, and etc",
                        default="mpistats_vs_gridshape"
                        )
    parser.add_argument('--dry-run',
                        action='store_true',
                        help="Simulate the process without making actual changes"
                        )

    args = parser.parse_args()
    perform_analysis(args)
