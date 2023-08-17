# Mini program for performance analysis of eigensolvers

## How to build

```bash
mkdir build
cd build

Cmake -C ../cmake/gcc.config.cmake ../.
Cmake --build .
Cmake --install . --prefix /path/to/install/dir
```

## How to use mpi4py

mpi4py should be installed using the same `mpi` library that is used to run the program

```bash
# openmpi
python -m venv venv-omp
source venv-omp/bin/activate
pip cache remove mpi4py
pip uninstall mpi4py
env MPICC=/path/to/omp/mpicc pip install mpi4py

# intel
python -m venv venv-intel
source venv-intel/bin/activate
pip cache remove mpi4py
pip uninstall mpi4py
env MPICC="/path/to/omp/mpiicc -cc=icx" pip install mpi4py
```
