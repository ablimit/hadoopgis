all: polytest deviceQuery

NVCC 		= /usr/local/cuda/bin/nvcc -m64
CC			= g++


UNAME 		= $(shell uname)
OPTFLAGS    = -O2
INCFLAGS    = -I/usr/local/include -I.
CFLAGS      = $(OPTFLAGS) $(INCFLAGS) 
LIBS        = -L/usr/local/lib -L.

ifeq ($(UNAME), Darwin)
CUDALIBS    = -lcudart
LDFLAGS		= -shared -flat_namespace -undefined suppress
else 
CUDALIBS    = -L/usr/local/cuda/lib64 -lcudart
endif
HFILES		= spatial.cuh 

deviceQuery: devicequery.cu
	$(NVCC) $^ -o $@

spatial.o: spatial.cu $(HFILES)
	$(NVCC) --ptxas-options=-v --compiler-options '-Wall' -c $*.cu

#gpu_lib: gpu-api.o poly.o gpu-info.o gpu-poly.o util.o
#	$(CC) $(LDFLAGS) -o gpu_spatial.so gpu-api.o poly.o gpu-info.o util.o gpu-poly.o $(CUDALIBS) $(LIBS)
	
# libspatial.so : spatial.o
#	$(NVCC) $(LDFLAGS) --ptxas-options=-v --compiler-options '-fPIC -Wall' --shared -o $@ $< $(CUDALIBS) $(LIBS)

polytest: driver.c spatial.o $(HFILES)
	$(CC) $< -o $@ $(CFLAGS) spatial.o $(CUDALIBS) $(LIBS)
clean:
	rm -f *.o deviceQuery polytest
	
