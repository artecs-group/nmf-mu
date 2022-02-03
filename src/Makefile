OPTIMIZE    = yes
DEBUG       = no
DEVICE      = cpu # {cpu, nvidia, igpu}
REAL        = simple # {simple, double}

CPU = CPU_DEVICE
IGPU = INTEL_GPU_DEVICE
NVIDIA = NVIDIA_DEVICE

# Program flags
PDEVICE = $(CPU)
PREAL = REAL_S

ifeq ($(DEVICE),nvidia)
  PDEVICE = NVIDIA_DEVICE
else ifeq ($(DEVICE), igpu)
  PDEVICE = $(IGPU)
endif

ifeq ($(REAL),double)
  PREAL = REAL_D
endif

# Default compiler and flags
CC = dpcpp
CFLAGS = -D$(PDEVICE) -D$(PREAL) -qmkl -DMKL_ILP64 -I"${MKLROOT}/include"
CLINK = -fsycl-device-code-split=per_kernel  ${MKLROOT}/lib/intel64/libmkl_sycl.a -Wl,-export-dynamic -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L${TBBROOT}/lib/intel64/gcc4.8 -ltbb -lsycl -lOpenCL -lpthread -lm -ldl

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

main: main.o ./device/device.o ./nmf/nmf.o ./common.o
	$(CC) $(CFLAGS) $(CLINK) main.o ./device/device.o ./nmf/nmf.o ./common.o -o main

main.o: ./device/device.o ./nmf/nmf.o main.cpp ./common.o
	$(CC) $(CFLAGS) main.cpp -c -o main.o

./nmf/nmf.o: ./nmf/nmf.cpp ./nmf/nmf.hpp ./common.o
	$(CC) $(CFLAGS) ./nmf/nmf.cpp -c -o ./nmf/nmf.o

./device/device.o: ./device/device.cpp ./device/device.hpp ./common.o
	$(CC) $(CFLAGS) ./device/device.cpp -c -o ./device/device.o

./common.o: ./common.hpp
	$(CC) $(CFLAGS) ./common.cpp -c -o ./common.o


run: main
	./main V.bin 100 100 4

.PHONY: clean
clean:
	rm -f main \
  *.o \
  ./nmf/nmf.o \
  ./device/device.o
