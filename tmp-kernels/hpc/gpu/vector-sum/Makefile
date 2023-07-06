CXX=g++
NVCC=nvcc
NVCCFLAGS = 
NVPROF=nvprof
INCS = -I../.

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCS) -o $@.x $<

# cu :  timer.cpp vectorAdd.cu
# 	# $(CC) $< -o $@
# 	$(NVCC) $^ -o vectorAdd.cu.x -I./

# cpp : vectorAdd.cpp 
# 	$(CXX) $^ -o vectorAdd.cpp.x -I./

# prof:
# 	$(NVPROF) ./vectorAdd.cu.x

clean:
	rm *.x