#KERNEL_VERSION=pixelgrid
KERNEL_VERSION=boxpixel

LIB_NAME=libgpuspatial.so

# DEBUG_OPT = -g -G

OBJECTS = parser.o util.o clip.o

all: $(OBJECTS)

# $(info Specify actions [ grid | box | install | clean])


grid: pixelgrid.cu gpu_spatial.h
	nvcc $(DEBUG_OPT) -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC -shared $< -o $(LIB_NAME)

box: boxpixel.cu gpu_spatial.h
	nvcc $(DEBUG_OPT) -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC -shared $< -o $(LIB_NAME)

parser.o : parser.cu parser.cuh
	nvcc $(DEBUG_OPT) -c -arch=sm_20 -O --ptxas-options=-v $< -o $@

util.o : util.cu util.cuh
	nvcc $(DEBUG_OPT) -c -arch=sm_20 -O --ptxas-options=-v $< -o $@

clip.o: boxpixel.cu gpu_spatial.h
	nvcc $(DEBUG_OPT) -c -arch=sm_20 -O --ptxas-options=-v $< -o $@

install:
	cp $(LIB_NAME) ~/softs/lib/

clean:
	rm -f *.o libgpuspatial.so
