# UAV Path Planning using ACO
A parallel version of Ant Colony Optimization (ACO) algorithm to solve Travelling Salesman Problem (TSP) on Graphics Processing Unit (GPU) by using Compute Unified Device Architecture (CUDA) platform.

## Getting Started

### Prerequisites
1. GPU device should be installed and driver should be configured
2. CUDA toolkit with library should be configured

### Pre-Compilation
1. Copy folders "__src__" and "__release__" including sub-folders at certian path e.g; /usr/share/aco/
2. Edit "__<instllation path>/release/makefile.init__" and change the __CUDA_PATH__ to the installed CUDA toolkit home
```
CUDA_PATH       ?= /usr/local/cuda-6.0
```

### Compilation settings for GPU
1. In the "__<instllation path>/release/makefile.init__" file, make sure this compilation flag to "__gpu__"
```
PLATFORM=gpu
```

### Compilation settings for CPU
1. In the "__<instllation path>/release/makefile.init__" file, make sure this compilation flag to "__cpu__"
```
PLATFORM=cpu
```

### Compiling
From the directory "__<instllation path>/release/__" execute command "__make__" once for "__CPU__" settings and once for "__GPU__" settings. 
This will create two executables as "__acouav_cpu__" and "__acouav_gpu__", repectively.

### Executing
1. Set any required parameters in shell script "__<instllation path>/release/acouav.sh__", like "__ACOUAV_HOME__" and "__MAXTIME__" to execute an iteration.
2. Preferably, go to the execution path "cd <instllation path>/release"
3. Execute the shell script by command "__./acouav.sh__"

### Collecting Results
After completion of the trials, the resulting stats will be in the "__<instllation path>/output__"







