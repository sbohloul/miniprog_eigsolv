CXX=g++
NVCC=nvcc
NVPROF=nvprof

cu :  timer.cpp vectorAdd.cu
	# $(CC) $< -o $@
	$(NVCC) $^ -o vectorAdd.cu.x -I./

cpp : timer.cpp vectorAdd.cpp 
	$(CXX) $^ -o vectorAdd.cpp.x -I./

prof:
	$(NVPROF) ./vectorAdd.cu.x

clean:
	rm *.x