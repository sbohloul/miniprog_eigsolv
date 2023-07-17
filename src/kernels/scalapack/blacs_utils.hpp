#ifndef _BLACS_UTILS_HPP_
#define _BLACS_UTILS_HPP_

class BlacsGrid
{
public:
    BlacsGrid(int, char *, int, int);

    int get_nprocs() const { return nprocs_; }
    int get_nprow() const { return nprow_; }
    int get_npcol() const { return npcol_; }
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
};

#endif