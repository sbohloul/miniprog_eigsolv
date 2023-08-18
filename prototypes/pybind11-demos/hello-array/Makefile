CXXOMP = mpic++ 
CXXINTEL = mpiicpc
CXXFLAGS = -Wall -std=c++11 -fPIC -shared

INTELLDFLAGS = -L${MKLROOT}/lib/intel64 
INTELLDFLAGS += -lmkl_scalapack_lp64
INTELLDFLAGS += -lmkl_blacs_intelmpi_lp64
INTELLDFLAGS += -lmkl_intel_lp64 
INTELLDFLAGS += -lmkl_sequential 
INTELLDFLAGS += -lmkl_core 
INTELLDFLAGS += -lpthread 
INTELLDFLAGS += -lm 
# LDFLAGS += -ldl

PYMODSUFFIX = $(shell python3-config --extension-suffix)

INCS = -I./
INCS += $(shell python3 -m pybind11 --includes)
# INCS += -I$(shell python -c "import mpi4py as m; print(m.get_include())")

SRC = pb11_module_array.cpp \
	array.cpp

OUT = _pb11_hello_array$(PYMODSUFFIX)

omp :
	$(CXXOMP) $(CXXFLAGS) $(INCS) $(SRC) -o $(OUT) 
	
intel:
	$(CXXINTEL) -cxx=icpx $(CXXFLAGS) $(INCS) $(SRC) -o $(OUT)