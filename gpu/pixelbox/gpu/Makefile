#KERNEL_VERSION=pixelgrid
KERNEL_VERSION=boxpixel

all:
	nvcc -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC -shared parser.cu util.cu ${KERNEL_VERSION}.cu -o libgpuspatial.so
clean:
	rm -f *.o libgpuspatial.so
