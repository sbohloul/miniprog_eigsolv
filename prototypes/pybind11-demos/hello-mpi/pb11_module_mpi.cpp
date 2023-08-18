#include <pybind11/pybind11.h>
#include <mpi4py/mpi4py.h>
#include "mpi_hello.hpp"

namespace py = pybind11;

MPI_Comm *get_mpi_comm(py::object py_comm)
{
    auto comm_ptr = PyMPIComm_Get(py_comm.ptr());

    if (!comm_ptr)
    {
        throw py::error_already_set();
    }

    return comm_ptr;
}

PYBIND11_MODULE(mpi_hello, m)
{

    // initialize mpi4py's C-API
    if (import_mpi4py() < 0)
    {
        // mpi4py calls the Python C API
        // we let pybind11 give us the detailed traceback
        throw py::error_already_set();
    }

    // m.doc() = R"pbdoc(
    //     Pybind11-mpi4py example plugin
    //     ------------------------------
    //     .. currentmodule:: _pb11mpi
    //     .. autosummary::
    //        :toctree: _generate
    //        greetings
    // )pbdoc";

    m.doc() = "pybind11 mpi hello plugin";

    m.def(
        "mpi_print_info",
        [](py::object py_comm)
        {
            auto comm = get_mpi_comm(py_comm);
            mpi_print_info(*comm);
        },
        "print mpi info");

    // m.def("mpi_print_info", &mpi_print_info, "Print mpi general info.");
    // m.def("mpi_my_rank", &mpi_my_rank, "Get mpi rank.");
}
