TBB_LIB_PATH=/home/kaibo/projects/gpu-db/tbb/tbb30_20110315oss/build/linux_intel64_gcc_cc4.4.6_libc2.12_kernel2.6.32_release
TBB_INC_PATH=/home/kaibo/projects/gpu-db/tbb/tbb30_20110315oss/include

CC= g++
CFLAGS= -c -O2 -I${TBB_INC_PATH} #-Wall
LDFLAGS= -L${TBB_LIB_PATH} -ltbb -Wl,-rpath,${TBB_LIB_PATH} -L./gpu -lgpuspatial -Wl,-rpath,./gpu -L./cpu -lcpuspatial -Wl,-rpath,./cpu
SOURCES= main.cpp constants.cpp parser.cpp builder.cpp filter.cpp aggregator.cpp spatial.c spatialindex.c hilbert.c rtree.c
OBJECTS= $(SOURCES:.cpp=.o)
EXECUTABLE= pipeline

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o $(EXECUTABLE)

