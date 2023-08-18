CXX = mpiicpc
CXXFLAGS = -Wall -std=c++11 -diag-disable=10441 -g -traceback -pedantic

INCS = 
INCS += -I./ 
INCS += -I/home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/utils
INCS += -I/home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/interfaces

LDFLAGS = -L${MKLROOT}/lib/intel64 
LDFLAGS += -lmkl_scalapack_lp64
LDFLAGS += -lmkl_blacs_intelmpi_lp64
LDFLAGS += -lmkl_intel_lp64 
LDFLAGS += -lmkl_sequential 
LDFLAGS += -lmkl_core 
LDFLAGS += -lpthread 
LDFLAGS += -lm 
# LDFLAGS += -ldl

UTILS = /home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/utils/timer.cpp

%.x : %.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)	

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ -c $< 

test%.x : test%.o
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ blacs_utils.o $< $(LDFLAGS)

test%.o : test%.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c blacs_utils.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c $<

clean:
	rm -rf *.o *.x
	