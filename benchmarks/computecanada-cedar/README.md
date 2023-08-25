# Benchmarks on Cedar

## Builds

- `mpi4py` should be loaded as a module (no need to install). Compatible version with `intelmpi` is `mpi4py/3.0.3` which works with `python/3.8`
- `flexiblas` is used for building with `gcc` and `openmpi`

### Build 1

#### Config summary

- intel compiler
- intel mpi
- mkl sequential

#### Setting the environment

Module are loaded by `source setenv-intel.sh`

```bash
module --force purge
module load StdEnv/2020  
module load intel/2020.1.217  
module load intelmpi/2019.7.217
module load imkl/2022.1.0
module load python/3.8
module load mpi4py/3.0.3
module load cmake/3.23.1
```

Python packages are installed as follows

```bash
python -m venv venv-intel
source venv-intel/bin/activate
pip install --upgrade pip
# numpy
pip install numpy
# pybind11
pip install pybind11
```

#### Compile and install

Use `ceadr.intel.sequential.config.cmake` config file to set build parameters:

```bash
source setenv-intel.sh
source venv-intel
mkdir build-intel-sequential && cd build-intel-sequential
cmake -C ../cmake/cedar.intel.sequential.config.cmake ../.
cmake --build .
cmake --install . --prefix ./
```

### Build 2

- intel compiler
- intel mpi
- mkl threaded

Similar to [build 1](#build-1) using `cedar.intel.threaded.config.cmake`.

### Build 3

#### Config summary

- gcc
- openmpi
- netlib scalapack
- flexiblas

#### Setting the environment

Module are loaded by `source setenv-gcc.sh`

```bash
module --force purge
module load StdEnv/2020
module load gcc/11.3.0
module load openmpi/4.1.4
module load scalapack/2.2.0
module load mpi4py/3.1.4
module load python/3.11.2
module load cmake/3.23.1
```

Python packages are installed as follows

```bash
python -m venv venv-gcc
source venv-gcc/bin/activate
pip install --upgrade pip
# numpy
pip install numpy
# pybind11
pip install pybind11
```

#### Compile and install

Use `cedar.gcc.ompi.netlibsca.fblas.config.cmake` config file to set build parameters:

```bash
source setenv-intel.sh
source venv-intel
mkdir build-intel-sequential && cd build-intel-sequential
cmake -C ../cmake/cedar.gcc.ompi.netlibsca.fblas.config.cmake ../.
cmake --build .
cmake --install . --prefix ./
```
