CXXOMP = mpic++ 
CXXINTEL = mpiicpc
CXXFLAGS = -Wall -std=c++11 -fPIC -shared

PYMODSUFFIX = $(shell python3-config --extension-suffix)

INCS = -I./
INCS += $(shell python3 -m pybind11 --includes)
INCS += -I $(shell python -c "import mpi4py as m; print(m.get_include())")

omp :
	$(CXXOMP) $(CXXFLAGS) $(INCS) -o mpi_hello$(PYMODSUFFIX) \
	mpi_hello.cpp pb11_module_mpi.cpp

intel:
	$(CXXINTEL) -cxx=icpx $(CXXFLAGS) $(INCS) -o mpi_hello$(PYMODSUFFIX) \
	mpi_hello.cpp pb11_module_mpi.cpp
