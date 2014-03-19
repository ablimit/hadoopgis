CC= gcc
CXX= g++
CFLAGS= -c -O2 -Wall -std=c++11 -fpermissive
# CXXFLAGS= -c -O2 -Wall -std=c++11
LDFLAGS= -L ~/softs/lib
#SOURCES= reducer.c spatialindex.c spatial.c

GPUEXECUTABLE = gpureducer
CPUEXECUTABLE = cpureducer
OBJECTS = rtree.o hilbert.o rstar.o spatial.o spatialindex.o crossmatch.o

gpu: $(GPUEXECUTABLE)

cpu: $(CPUEXECUTABLE)

all: cpu gpu

debug: CXX += -pg
debug: CC += -pg
debug: all

$(GPUEXECUTABLE): $(OBJECTS)
	$(CXX) -DGPU $(LDFLAGS) -lgpuspatial $(OBJECTS) -o $@
# $(CXX) $(LDFLAGS) $(OBJECTS) -Wl,-rpath,./cuda -o $@

$(CPUEXECUTABLE): $(OBJECTS)
	$(CXX) -DCPU $(LDFLAGS) -lcpuspatial $(OBJECTS) -o $@

crossmatch.o: reducer.f.cpp crossmatch.h
	$(CXX) $(CFLAGS) $< -o $@

spatial.o: spatial.c spatial.h
	$(CXX) $(CFLAGS) $< -o $@

spatialindex.o: spatialindex.c spatialindex.h
	$(CXX) $(CFLAGS) $< -o $@

rtree.o: rtree.c rtree.h
	$(CXX) $(CFLAGS) $< -o $@

rstar.o: rstar.c rstar.h
	$(CXX) $(CFLAGS) $< -o $@

hilbert.o: hilbert.c hilbert.h
	$(CXX) $(CFLAGS) $< -o $@

clean:
	rm -f *.o $(GPUEXECUTABLE) $(CPUEXECUTABLE)

