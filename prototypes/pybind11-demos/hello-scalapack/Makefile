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

SCALAPACKROOT = /home/sbohloul/.local/scalapack/2.1.0
OPENBALSROO = /home/sbohloul/.local/openblas/0.3.21/lib
OMPLDFLAGS =
OMPLDFLAGS += -L${SCALAPACKROOT} -lscalapack
OMPLDFLAGS += -L${OPENBALSROO} -lopenblas
OMPLDFLAGS += -lgfortran


PYMODSUFFIX = $(shell python3-config --extension-suffix)

INCS = -I./
INCS += $(shell python3 -m pybind11 --includes)
INCS += -I$(shell python -c "import mpi4py as m; print(m.get_include())")

SRC = pb11_module_time_scalapack_kernels.cpp \
	pb11_time_scalapack_kernels.cpp \
	time_scalapack_kernels.cpp \
	timer.cpp \
	array_helper.cpp

OUT = _pb11_time_scalapack_kernels

omp :
	$(CXXOMP) $(CXXFLAGS) $(INCS) $(SRC) -o $(OUT)$(PYMODSUFFIX) $(OMPLDFLAGS)
	
intel:
	$(CXXINTEL) -cxx=icpx $(CXXFLAGS) $(INCS) $(SRC) -o $(OUT)$(PYMODSUFFIX)  $(INTELLDFLAGS)

pure:
	mpic++ -Wall -std=c++11 \
	-I./ \
	driver_time_scalapack_pdgemm.cpp \
	time_scalapack_kernels.cpp \
	timer.cpp \
	array_helper.cpp \
	-o prog.x \
	-L/home/sbohloul/.local/scalapack/2.1.0 -lscalapack	\
	-L/home/sbohloul/.local/openblas/0.3.21/lib -lopenblas \
	-lgfortran