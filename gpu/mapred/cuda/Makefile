KERNEL_VERSION=pixelgrid

all:
	nvcc -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC -shared ${KERNEL_VERSION}.cu -o libgpuspatial.so
clean:
	rm -f *.o libgpuspatial.so
