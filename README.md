# CuQuEmBench: CUDA Quantum Emulator Benchmark
Quantum Phase Estimation (QPE) emulator on NVIDIA GPU

## Before build

1. Install the followings
- [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk)
- [cuQuantum](https://developer.nvidia.com/cuquantum-sdk)
- [HDF5](https://www.hdfgroup.org/download-hdf5/)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp): version 0.8.0 or higher

2. Set the environment variables
- "HDF5_ROOT" : path to cmake configuration files of HDF5 library
- "CMAKE_PREFIX_PATH" : so this var includes path to cmake configuration files of yaml-cpp
- "CUQUANTUM_ROOT" : path to root directory of installed cuQuantum library (must contain lib and include dirs)

3. Prepare python environment for pre/post process with following libraries
- h5py
- qiskit
- matplotlib
- pyyaml

## Build and execute example
```bash
git clone <this repository>
cd <this repository dir>

python3 -m venv ./venv
. ./venv/bin/activate

pip install qiskit h5py matplotlib pyyaml

mkdir build && cd build

export HDF5_ROOT=path/to/hdf5/cmake/config/files
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:+${CMAKE_PREFIX_PATH}:}path/to/yaml-cpp/cmake/config/files

cmake ..
make

cp ../example/config.yaml ./

cp ../python_prepost/prepare_unitary_matrix.py ./
python prepare_unitary_matrix.py

./qpe.out

cp ../python_prepost/draw_statevector.py ./
python draw_state_vector.py

```

