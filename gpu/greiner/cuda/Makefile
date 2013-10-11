all: gpu_lib

NVCC 		= nvcc -m64
CC			= gcc


UNAME 		= $(shell uname)
OPTFLAGS    = 
INCFLAGS    = -I/usr/local/pgsql/include/server -I/usr/local/include
CFLAGS      = $(OPTFLAGS) $(INCFLAGS) 
LIBS        = -L/usr/local/lib

ifeq ($(UNAME), Darwin)
CUDALIBS    = -L/usr/local/cuda/lib -lcudart
LDFLAGS		= -shared -flat_namespace -undefined suppress
else 
CUDALIBS    = -L/opt/local/cuda/lib64 -lcudart
endif
HFILES		= spatial.cuh 

devq: devicequery.cu
	$(NVCC) $^ -o deviceQuery

spatial.o: spatial.cu $(HFILES)
	$(NVCC) --ptxas-options=-v --compiler-options '-fPIC -Wall' -c $*.cu

#gpu_lib: gpu-api.o poly.o gpu-info.o gpu-poly.o util.o
#	$(CC) $(LDFLAGS) -o gpu_spatial.so gpu-api.o poly.o gpu-info.o util.o gpu-poly.o $(CUDALIBS) $(LIBS)
	
gpu_lib: spatial.o
	$(NVCC) $(LDFLAGS) --ptxas-options=-v --compiler-options '-fPIC -Wall' --shared -o gpuspatial.so $< $(CUDALIBS) $(LIBS)
clean:
	rm -f *.o deviceQuery *.so
	
