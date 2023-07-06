CXX	= g++
CXXFLAGS = -pthread -fopenmp
INCS = -I./

SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
EXES = $(SRCS:.cpp=.x)

all: $(EXES)

%.x: %.o
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c $< -o $@

clean:
	rm -f *.o *.x

