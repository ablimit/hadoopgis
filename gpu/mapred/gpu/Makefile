#KERNEL_VERSION=pixelgrid
KERNEL_VERSION=boxpixel

LIB_NAME=libgpuspatial.so

# DEBUG_OPT = -g -G

all: 
	$(info Specify actions [ grid | box | install | clean])

grid: pixelgrid.cu gpu_spatial.h parser.cu util.cu
	nvcc $(DEBUG_OPT) -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC -shared parser.cu util.cu $< -o $(LIB_NAME)

box: boxpixel.cu gpu_spatial.h parser.cu util.cu
	nvcc -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC -shared parser.cu util.cu $< -o $(LIB_NAME)

install:
	cp $(LIB_NAME) ~/softs/lib/

clean:
	rm -f *.o libgpuspatial.so
