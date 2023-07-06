NVCC = nvcc
NVCCFLAGS =
CXX = g++
CXXFLAGS = 
INCS = -I/home/sbohloul/Projects/scicomp/miniprog_eigsolv/utils-timing

SRCS = $(wildcard *.cu)
OBJS = $(SRCS:.cu=.o)
EXES = $(SRCS:.cu=.x)

all: $(EXES)

%.x: %.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^


%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCS) -c $<

clean:
	rm -rf *.o *.x

# run.%:
# 	./$*.x

# run:
# 	# ./$(EXES)
# 	for f in $(EXES); do ./$$f; done
