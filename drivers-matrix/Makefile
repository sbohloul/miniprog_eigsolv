SRCS = $(wildcard *.cpp)
OBJS = $(SRCS:.cpp=.o)
EXES = $(SRCS:.cpp=.x)

CXX = g++
CXXFLAGS = -g

INCS = -I./../utils-matrix

all: $(EXES)

%.x: %.o
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c $< 

clean:
	rm -rf *.o *.x

run:
	# ./$(EXES)
	for f in $(EXES); do ./$$f; done

run.%:
	./$*.x