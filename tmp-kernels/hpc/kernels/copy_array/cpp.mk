CXX = g++
CXXFLAGS = -O3 -march=native
INCS = -I/home/sbohloul/Projects/scicomp/miniprog_eigsolv/utils-timing

SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
EXES = $(SRCS:.cpp=.x)

all: $(EXES)

%.x: %.o
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c $<

clean:
	rm -rf *.o *.x