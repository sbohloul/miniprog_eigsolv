CXX = mpicxx
CXXFLAGS = -Wall -std=c++11 -g

INCS = 
INCS += -I./ 
INCS += -I/home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/utils
INCS += -I/home/sbohloul/Projects/scicomp/miniprog_eigsolv/src/interfaces

SCALAPACK_DIR = /home/sbohloul/.local/scalapack/2.2.0
LDFLAGS = -L${SCALAPACK_DIR} 
LDFLAGS += -lscalapack -lgfortran

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
	