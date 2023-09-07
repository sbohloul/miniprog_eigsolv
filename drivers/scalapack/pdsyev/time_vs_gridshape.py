from mpi4py import MPI
import pb11_time_scalapack_kernels as pb11tsk
import numpy as np
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
    MB_MIN = 8
    NB_MIN = 8

    dry_run = args.dry_run
    outfile = args.output + ".csv"
    outplot = args.output + ".svg"
    niter = args.niter
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
            print(f"niter: {niter} \n"
                  f"mb_max: {mb_max}, nb_max: {nb_max}\n"
                  f"mb: {mb_list}\n"
                  f"nb: {nb_list}\n"
                  f"inputfile: {fname}\n"
                  f"proc_grid: {proc_grid}"
                  )
        return

    # ============
    # Input arrays
    # ============
    if (mpirank == 0):
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
            nprow = p
            npcol = q
            t_tmp = pb11tsk.pb11_time_scalapack_pdsyev(
                niter, nprow, npcol, a, eigval, eigvec, m, mb, nb)
            tkernel.append(t_tmp)
        # Gather all timings in rank 0
        tkernel = mpicomm.gather(tkernel, root=0)
        # [(mb, nb), [[rank0], [rank1], ...]]
        data_all.append([[mb, nb], tkernel])

    # ==============
    # Output results
    # ==============
    # write data to output
    index_tuple = []
    if mpirank == 0:
        df = pd.DataFrame()
        for data_blk in data_all:
            mb, nb = data_blk[0]
            tkernel = data_blk[1]
            # update tuple for multi-index
            for i in range(nprocs):
                index_tuple.append(((mb, nb), i))
            #
            current_df = pd.DataFrame(
                tkernel, columns=[(p, q) for p, q in proc_grid])
            #
            df = pd.concat([df, current_df], ignore_index=False)

        # df.index.name = 'rank'
        index = pd.MultiIndex.from_tuples(index_tuple,
                                          names=['blocksize', 'rank'])
        df.set_index(index, inplace=True)
        df.columns.name = 'processgrid'
        print(df)
        with open(outfile, 'w') as f:
            f.write(
                "# time vs process grid shape\n"
                f"# nprocs: {nprocs}\n"
                f"# mb: {mb_list} , nb: {nb_list}\n"
                f"# mb_max: {mb_max}, nb_max: {nb_max}\n"
                f"# m: {m}\n"
                f"# process grid: {proc_grid}\n"
                f"\n"
            )
            df.to_csv(f, sep='\t', index=True,
                      header=True, float_format='%.6f')

    # Plot time vs block size
    if mpirank == 0:
        xdata = [i+1 for i in range(len(proc_grid))]

        xlabel = "p"
        ylabel = "q"
        zlabel = "Time (s)"
        title = f"pdsyev \n mb: {mb}, nb: {nb} \n m: {m}"

        nfigcols = 2
        nfigrows = len(data_all) // nfigcols + 1
        fig, axs = plt.subplots(
            nfigrows, nfigcols
        )
        fig.suptitle(title)
        axs = axs.flatten()
        for iax, data_blk in enumerate(data_all):
            mb, nb = data_blk[0]
            tkernel = data_blk[1]
            ax = axs[iax]
            for iproc in range(nprocs):
                zdata = tkernel[iproc]
                ax.plot(
                    xdata, zdata,
                    label="rank " + str(iproc),
                    marker='o'
                )

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                # ax.set_zlabel(zlabel)
                ax.grid()
                # ax.legend()
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
        - Measure timing for all possible p x q grids for a given nprocs
        - mb = [64, 128, 256, ..., max_mb]
        - nb = [64, 128, 256, ..., max_nb]
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "niter", type=int, help="Number of iterations used for averaging the timing")
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
                        default="time_vs_gridshape"
                        )
    parser.add_argument('--dry-run',
                        action='store_true',
                        help="Simulate the process without making actual changes"
                        )

    args = parser.parse_args()
    perform_analysis(args)
