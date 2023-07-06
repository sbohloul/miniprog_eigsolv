default: kernel

# .PHONY:	kernel

LIKWID_PATH=/home/sbohloul/.local/likwid/5.2.2
LDLIBS=-L${LIKWID_PATH}/lib -llikwid

CXX=g++
CXXFLAGS=-g -O3 -march=native -ftree-vectorize -I./ -I${LIKWID_PATH}/include -DLIKWID_PERFMON

kernel: kernel.o timer.o
	$(CXX) -o $@ $? ${LDLIBS}

%.o:%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f kernel *.o