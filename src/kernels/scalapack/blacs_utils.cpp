#include <iostream>
#include "blacs_utils.hpp"
#include "blacs.h"

BlacsGrid::BlacsGrid(int ictxt, char *order, int nprow, int npcol) : ctxt_(ictxt), order_(order), nprow_(nprow), npcol_(npcol)
{
    Cblacs_pinfo(&myrank_, &nprocs_);
    Cblacs_gridinit(&ctxt_, order_, nprow_, npcol_);
    Cblacs_gridinfo(ctxt_, &nprow_, &npcol_, &myrow_, &mycol_);
}

BlacsArrayDescription::BlacsArrayDescription(BlacsGrid *grid, int m, int n, int mb, int nb) : grid_(grid)
{
    // desc_A(1) = 1            only possible value for the descriptor type
    // desc_A(2) = ctxt_A       BLACS context for the distribution of the global matrix A
    // desc_A(3) = m_A          number of rows of global matrix A
    // desc_A(4) = n_A          number of columns of global matrix A
    // desc_A(5) = mb_A         column block size for the distribution of A
    // desc_A(6) = nb_A         row block size for the distribution of A
    // desc_A(7) = rsrc_A       process row in the ctxt_A grid containing the first row of A
    // desc_A(8) = csrc_A       process column in the ctxt_A grid containing the first column of A
    // desc_A(9) = llda         leading dimension of the local matrix containing elements of A

    int ctxt = grid_->get_context();
    if (~ctxt < 0)
    {
        int nprow = grid_->get_nprow();
        int npcol = grid_->get_npcol();
        int myrow = grid_->get_myrow();
        int mycol = grid_->get_mycol();
        int rsrcproc = grid_->get_row_srcproc();
        int csrcproc = grid_->get_col_srcproc();

        ml_ = numroc(&m, &mb, &myrow, &rsrcproc, &nprow);
        nl_ = numroc(&n, &nb, &mycol, &csrcproc, &npcol);
        int lld = std::max(1, ml_);

        descinit(desc_, &m, &n, &mb, &nb, &rsrc_, &csrc_, &ctxt, &lld, &info_);
    }
    else
    {
        ml_ = 0;
        nl_ = 0;
        info_ = -20;

        for (int i = 0; i < 9; i++)
        {
            desc_[i] = 0;
        }
        desc_[1] = -1;
    }
}

void BlacsArrayDescription::get_desc(int *desc) const
{
    for (int i = 0; i < 9; i++)
    {
        desc[i] = desc_[i];
    }
}

//
DistributedArray::DistributedArray(BlacsArrayDescription *desc) : desc_(desc), data_(desc_->local_size_row() * desc_->local_size_col()) {}