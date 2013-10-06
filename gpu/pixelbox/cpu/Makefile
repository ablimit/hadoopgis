# NOTE: you have to install cilk++ on your machine in order to compile and use this
# library. Dowload cilk++ from http://software.intel.com/en-us/articles/download-intel-cilk-sdk/
# Before make, set CILK_INSTALL_PATH to where you installed cilk++

TBB_LIB_PATH=/home/kaibo/projects/gpu-db/tbb/tbb30_20110315oss/build/linux_intel64_gcc_cc4.4.6_libc2.12_kernel2.6.32_release
TBB_INC_PATH=/home/kaibo/projects/gpu-db/tbb/tbb30_20110315oss/include

CC= g++
CFLAGS= -O2 -I${TBB_INC_PATH} -Wall -fPIC -shared
LDFLAGS= -L${TBB_LIB_PATH} -ltbb -Wl,-rpath,${TBB_LIB_PATH}

KERNEL_VERSION=pixelgrid

CC=g++

all:
	${CC} ${CFLAGS} ${LDFLAGS} ${KERNEL_VERSION}.cpp -o libcpuspatial.so
clean:
	rm -f *.o libcpuspatial.so
