CXX = mpiicpc
CXXFLAGS = -Wall -std=c++11 -diag-disable=10441 -g -traceback -pedantic

INCS = 
INCS += -I./ 
INCS += -I/home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/utils
INCS += -I/home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/interfaces

LDFLAGS = -L${MKLROOT}/lib/intel64 
LDFLAGS += -lmkl_scalapack_lp64
LDFLAGS += -lmkl_intel_lp64 
LDFLAGS += -lmkl_sequential 
LDFLAGS += -lmkl_core 
LDFLAGS += -lmkl_blacs_intelmpi_lp64
LDFLAGS += -lpthread 
LDFLAGS += -lm 
# LDFLAGS += -ldl

UTILS = /home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/utils/timer.cpp

%.x : %.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)	

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ -c $< 


test_blacs_utils.x :
	$(CXX) $(CXXFLAGS) $(INCS) -c blacs_utils.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c test_blacs_utils.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ blacs_utils.o test_blacs_utils.o $(LDFLAGS)


test_blacs_hello.x :
	$(CXX) $(CXXFLAGS) $(INCS) -c blacs_utils.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c test_blacs_hello.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -o $@ blacs_utils.o test_blacs_hello.o $(LDFLAGS)




clean:
	rm -rf *.o *.x
	