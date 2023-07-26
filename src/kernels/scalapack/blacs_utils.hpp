#ifndef _BLACS_UTILS_HPP_
#define _BLACS_UTILS_HPP_

#include <vector>

class BlacsGrid
{
public:
    BlacsGrid(int ctxt, char *order, int nprow, int npcol);

    int get_nprocs() const { return nprocs_; }
    int get_nprow() const { return nprow_; }
    int get_npcol() const { return npcol_; }
    int get_row_srcproc() const { return rsrcproc_; }
    int get_col_srcproc() const { return csrcproc_; }
    int get_myrank() const { return myrank_; }
    int get_mycol() const { return mycol_; }
    int get_myrow() const { return myrow_; }
    int get_context() const { return ctxt_; }
    char *get_order() const { return order_; }

private:
    int ctxt_;
    char *order_;
    int nprow_;
    int npcol_;
    int nprocs_;
    int myrank_;
    int myrow_;
    int mycol_;

    int rsrcproc_ = 0;
    int csrcproc_ = 0;
};

class BlacsArrayDescription
{
public:
    BlacsArrayDescription(BlacsGrid *grid, int m, int n, int mb, int nb);

    void get_desc(int *desc) const;
    int get_desctype() const { return desc_[0]; }
    int get_context() const { return desc_[1]; }
    int global_size_row() const { return desc_[2]; }
    int global_size_col() const { return desc_[3]; }
    int block_size_row() const { return desc_[4]; }
    int block_size_col() const { return desc_[5]; }
    int local_size_row() const { return ml_; }
    int local_size_col() const { return nl_; }
    int get_row_src() const { return desc_[6]; }
    int get_col_src() const { return desc_[7]; }
    int get_lld() const { return desc_[8]; }
    int get_info() const { return info_; }
    BlacsGrid *grid() const { return grid_; }
    // BlacsGrid *get_gird() const { return grid_; }

private:
    int desc_[9];
    int info_;
    int ml_;
    int nl_;
    BlacsGrid *grid_;

    int rsrc_ = 0;
    int csrc_ = 0;
};

class DistributedArray
{
public:
    DistributedArray(BlacsArrayDescription *desc);

    BlacsArrayDescription *get_desc() const { return desc_; }
    BlacsGrid *get_grid() const { return desc_->grid(); }
    const double *data() const { return data_.empty() ? 0 : &data_.front(); }
    double *data() { return data_.empty() ? 0 : &data_.front(); }

    int size() { return data_.size(); }
    int get_nrow() { return desc_->local_size_row(); }
    int get_ncol() { return desc_->local_size_col(); }
    int get_lld() { return desc_->get_lld(); }

private:
    BlacsArrayDescription *desc_;
    std::vector<double> data_;
};

#endif