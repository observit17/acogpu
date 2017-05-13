################################################################################
# 
################################################################################



-include makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir.mk
-include src/subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(CU_DEPS)),)
-include $(CU_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: build


build: acouav

# Tool invocations
acouav: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@_$(PLATFORM)_$(AGING)_$(BROKER)'
	@echo 'Invoking: NVCC Linker'
	#$(CUDA_PATH)/bin/nvcc --cudart static -link -o  "acouav" $(OBJS) $(USER_OBJS) $(LIBS)
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@_$(PLATFORM)_$(AGING)_$(BROKER) $(LIBRARIES) $(OBJS) $(USER_OBJS)
	mkdir -p ./bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))
	cp $@_$(PLATFORM)_$(AGING)_$(BROKER) ./bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))
	@echo 'Finished building target: $@'
	@echo ' '

run: build
	./acouav_$(PLATFORM)_$(AGING)_$(BROKER)

# Other Targets
clean:
	-$(RM) $(CC_DEPS)$(C++_DEPS)$(EXECUTABLES)$(C_UPPER_DEPS)$(CXX_DEPS)$(OBJS)$(CU_DEPS)$(CPP_DEPS)$(C_DEPS)\
		acouav_$(PLATFORM)_$(AGING)_$(BROKER)\
		./bin/$(OS_ARCH)/$(OSLOWER)/$(TARGET)$(if $(abi),/$(abi))/acouav_$(PLATFORM)_$(AGING)_$(BROKER)
	-@echo ' '

clobber: clean

.PHONY: all clean dependents
.SECONDARY:

-include makefile.targets
