UAV path planning using ACO

Pre-requisites:
---------------
1- GPU device should be installed and driver should be configured
2- CUDA toolkit with library should be configured

Pre-Compilation:
----------------
1- Copy folders 'src' and 'release' including sub-folders at certian path e.g; /usr/share/aco/
2- Edit "<instllation path>/release/makefile.init" and change the CUDA_PATH to the installed CUDA toolkit home
	CUDA_PATH       ?= /usr/local/cuda-6.0

Compilation settings for GPU:
------------------
1- In the "<instllation path>/release/makefile.init" file, make sure this compilation flag to "gpu"
PLATFORM=gpu


Compilation settings for CPU:
------------------
1- In the "<instllation path>/release/makefile.init" file, make sure this compilation flag to "cpu"
PLATFORM=cpu

Compiling:
----------
From the directory "<instllation path>/release/" execute command "make" once for "CPU" settings and once for "GPU" settings. 
This will create two executables as "acouav_cpu" and "acouav_gpu", repectively.

Executing:
---------
1- Set any required parameters in shell script "<instllation path>/release/acouav.sh", like "ACOUAV_HOME" and "MAXTIME" to execute an iteration.
2- Preferably, go to the execution path "cd <instllation path>/release"
3- Execute the shell script by command "./acouav.sh"

Collecting Results:
-------------------
After completion of the trials, the resulting stats will be in the "<instllation path>/output"







