# Benchmarks on Niagara

## Builds

### Build 1

#### Config summary

- intel compiler
- intel mpi
- mkl sequential

```bash
$ module list

Currently Loaded Modules:
  1) NiaEnv/2019b   3) intelmpi/2019u3   5) cmake/3.21.4
  2) intel/2019u3   4) python/3.9.8
```

#### Setting the environment

Module are loaded by `source setenv-intel.sh`

```bash
module load intel/2019u3    # this will also load mkl
module load intelmpi/2019u3
module load python/3.9.8
module load cmake/3.21.4
```

Python packages are installed as follows, note that `numpy` should be installed as `intel-numpy` otherwise there will be issues related to loading sevela `mkl` libraries

```bash
python -m venv venv-intel
source venv-intel/bin/activate
pip install --upgrade pip
# numpy
pip install intel-numpy
# pybind11
pip install pybind11
# mpi4py
pip uninstall mpi4py
pip cache remove mpi4py # necessary to make sure it will be installed by correct mpi library
env MPICC=mpicc pip install mpi4py
```

#### Compile and install

Use `niagara.intel.sequential.config.cmake` config file to set build parameters:

```bash
source setenv-intel.sh
source venv-intel
mkdir build-intel-sequential && cd build-intel-sequential
cmake -C ../cmake/niagara.intel.sequential.config.cmake ../.
cmake --build .
cmake --install . --prefix ./
```

### Build 2

- intel compiler
- intel mpi
- mkl threaded

Similar to [build 1](#build-1) using `niagara.intel.threaded.config.cmake`.
