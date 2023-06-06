CXX = g++
CXXFLAGS = -std=c++11
SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
EXES = $(SRCS:.cpp=.x)

OPENBLAS_LIB = /home/sbohloul/.local/openblas/0.3.21/lib
OPENBLAS_INC = /home/sbohloul/.local/openblas/0.3.21/include

INCS = -I./../utils-matrix -I./../utils-blas

ifdef USE_MKL
	CXXFLAGS += -DUSE_MKL
	LIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
		   -lpthread -lm -ldl
	# INCS = -I$(MKLROOT)/include
	# INCS = -I./
else ifdef USE_OPENBLAS
	CXXFLAGS += -DUSE_OPENBLAS -static
	LIBS = -L$(OPENBLAS_LIB) -lopenblas
	# INCS = -I$(OPENBLAS_INC)
	# INCS = -I./
else
	CXXFLAGS += -static
	LIBS = -L$(OPENBLAS_LIB) -lopenblas
	# INCS = -I./
endif

all: $(EXES)

%.x: %.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c $< 

clean:
	rm -rf *.o *.x

run:
	for f in $(EXES); do ./$$f; done

run.%:
	./$*.x