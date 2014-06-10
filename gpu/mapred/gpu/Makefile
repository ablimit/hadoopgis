#KERNEL_VERSION=pixelgrid
KERNEL_VERSION=boxpixel

LIB_NAME=libgpurefine.so

# DEBUG_OPT = -g -G

INCFLAG=-I.. 
# CFLAG=-m64

OBJECTS = parser.o util.o clip.o

all: $(OBJECTS)
	nvcc $(DEBUG_OPT) $(CFLAG) -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC -shared $(INCFLAG) $(OBJECTS) -o $(LIB_NAME)


# $(info Specify actions [ grid | box | install | clean])


grid: pixelgrid.cu parser.o util.o
	nvcc $(DEBUG_OPT) $(CFLAG) -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC -shared $(INCFLAG) $< -o $(LIB_NAME)

box: boxpixel.cu gpu_refine.h
	nvcc $(DEBUG_OPT) $(CFLAG) -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC -shared $(INCFLAG) $< -o $(LIB_NAME)

parser.o : parser.cu parser.cuh
	nvcc $(DEBUG_OPT) $(CFLAG) -c  -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC $(INCFLAG) $< -o $@

util.o : util.cu util.cuh
	nvcc $(DEBUG_OPT) $(CFLAG) -c -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC $(INCFLAG) $< -o $@

clip.o: boxpixel.cu gpu_refine.h
	nvcc $(DEBUG_OPT) $(CFLAG) -c -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC $(INCFLAG) $< -o $@

install:
	cp $(LIB_NAME) ~/softs/lib/

clean:
	rm -f *.o $(LIB_NAME)
