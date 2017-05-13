################################################################################
# 
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/AntSystem.cu \
../src/Colony.cu \
../src/RankBasedAntSystem.cu \
../src/Utils.cu \
../src/TSPReader.cu 

CPP_SRCS += \
../src/Commandline.cpp \
../src/Setup.cpp \
../src/TSP.cpp \
../src/Timer.cpp \
../src/Writer.cpp 

OBJS += \
./src/AntSystem.o \
./src/Colony.o \
./src/Commandline.o \
./src/RankBasedAntSystem.o \
./src/Setup.o \
./src/TSP.o \
./src/TSPReader.o \
./src/Timer.o \
./src/Utils.o \
./src/Writer.o 

CU_DEPS += \
./src/AntSystem.d \
./src/Colony.d \
./src/RankBasedAntSystem.d \
./src/Utils.d \
./src/TSPReader.d 

CPP_DEPS += \
./src/Commandline.d \
./src/Setup.d \
./src/TSP.d \
./src/Timer.d \
./src/Writer.d 



# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	$(NVCC) $(INCLUDES) --compile -O3 $(GENCODE_FLAGS)  -x cu -o  "$@" "$<"
	#/usr/local/cuda-6.0/bin/nvcc -O3 -gencode arch=compute_10,code=sm_10  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	#/usr/local/cuda-6.0/bin/nvcc --compile -O3 -gencode arch=compute_10,code=sm_10  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	$(NVCC) $(INCLUDES) -O3 $(GENCODE_FLAGS) -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	$(NVCC) $(INCLUDES) -O3 --compile  -x c++ -o  "$@" "$<"
	#/usr/local/cuda-6.0/bin/nvcc -O3 -gencode arch=compute_10,code=sm_10  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	#/usr/local/cuda-6.0/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


