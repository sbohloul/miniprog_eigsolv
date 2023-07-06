CXX = g++
CXXFLAGS = -std=c++11 -fopenmp -pthread
INCS = -I../.

% : %.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -o $@.x $<

clean:
	rm -rf *.x *.o