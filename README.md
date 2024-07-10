# OpenMP Polymorphic Memory Resource
This is an example of a custom C++ polymorphic memory resource that uses OpenMP's offload device memory management functionality.

## Dev Env
On Ubuntu 22.04 with an Nvidia GPU:
- `sudo apt install gcc-11 gcc-11-offload-nvptx g++-11`

On Ubuntu 22.04 with an AMD GPU:
- `sudo apt install gcc-11 gcc-11-offload-amdgcn g++-11`

## Build
Using the GCC compilation suite with an Nvidia GPU:
- `g++-11 -std=c++17 -fopenmp -foffload=nvptx-none -o app.exe ompDeviceMemoryResource.cpp`

- Using the GCC compilation suite with an AMD GPU:
- `g++-11 -std=c++17 -fopenmp -foffload=amdgcn-amdhsa -o app.exe ompDeviceMemoryResource.cpp`
