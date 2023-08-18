#include "pb11_time_scalapack_kernels.hpp"

PYBIND11_MODULE(pb11_time_scalapack_kernels, m)
{

    // initialize mpi4py's C-API
    if (import_mpi4py() < 0)
    {
        // mpi4py calls the Python C API
        // we let pybind11 give us the detailed traceback
        throw py::error_already_set();
    }

    m.doc() = "Module for time scalapack kernels";

    // m.def(
    //     "time_scalapack_pdgemm",
    //     [](py::object py_comm)
    //     {
    //         auto comm = get_mpi_comm(py_comm);
    //         mpi_print_info(*comm);
    //     },
    //     "print mpi info");

    // m.def("mpi_print_info", &mpi_print_info, "Print mpi general info.");
    // m.def("mpi_my_rank", &mpi_my_rank, "Get mpi rank.");

    m.def("pb11_time_scalapack_pdgemm", &pb11_time_scalapack_pdgemm, "time pdgemm");
}