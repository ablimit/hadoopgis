CC= gcc
CXX= g++
CFLAGS= -c -O2 -Wall -std=c++0x -fpermissive
# CXXFLAGS= -c -O2 -Wall -std=c++11
ifneq (,$(findstring emory,$(shell uname -a)))
	LDFLAGS= -L ~/softs/lib
else 
	LDFLAGS= -L /usr/local/lib
endif
#SOURCES= reducer.c spatialindex.c spatial.c

GPUEXECUTABLE = gpureducer
CPUEXECUTABLE = cpureducer
OBJECTS = rtree.o hilbert.o rstar.o spatial.o spatialindex.o crossmatch.o

all: cpu

prof: PROFILE_FLAG = -pg
prof: all

gpu: DEVICE = GPU
gpu: $(GPUEXECUTABLE)
gpu: OBJECTS += gpu/parser.o gpu/util.o gpu/clip.o
gpu: CFLAGS += -I./gpu -I.


cpu: DEVICE = CPU
cpu: $(CPUEXECUTABLE)
cpu: CFLAGS += -I./cpu -I.


debug: CXX += -g
debug: CC += -g
debug: all


$(GPUEXECUTABLE): $(OBJECTS)
	nvcc -arch=sm_20 -O --ptxas-options=-v $(OBJECTS) -o $@

#$(CXX) $(PROFILE_FLAG) $(LDFLAGS) $(OBJECTS) -o $@
#$(CXX) $(LDFLAGS) $(OBJECTS) -Wl,-rpath,./cuda -o $@

$(CPUEXECUTABLE): $(OBJECTS)
	$(CXX) -D$(DEVICE) $(PROFILE_FLAG) $(LDFLAGS) -lcpurefine $(OBJECTS) -o $@

crossmatch.o: reducer.fgpu.parse.cpp crossmatch.h
	$(CXX) -D$(DEVICE) $(PROFILE_FLAG) $(CFLAGS) $< -o $@

spatial.o: spatial.c spatial.h
	$(CXX) $(PROFILE_FLAG) $(CFLAGS) $< -o $@

spatialindex.o: spatialindex.c spatialindex.h
	$(CXX) $(PROFILE_FLAG) $(CFLAGS) $< -o $@

rtree.o: rtree.c rtree.h
	$(CXX) $(PROFILE_FLAG) $(CFLAGS) $< -o $@

rstar.o: rstar.c rstar.h
	$(CXX) $(PROFILE_FLAG) $(CFLAGS) $< -o $@

hilbert.o: hilbert.c hilbert.h
	$(CXX) $(PROFILE_FLAG) $(CFLAGS) $< -o $@


clean:
	rm -f *.o $(GPUEXECUTABLE) $(CPUEXECUTABLE)

