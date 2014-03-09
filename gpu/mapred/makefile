CC= gcc
CXX= g++
CFLAGS= -c -O2 -Wall -fpermissive
CXXFLAGS= -c -O2 -Wall -std=c++11
LDFLAGS= -L./cuda -lgpuspatial
SOURCES= reducer.c spatialindex.c spatial.c

EXECUTABLE = cudareducer
OBJECTS = hilbert.o rstar.o spatial.o spatialindex.o crossmatch.o

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) -Wl,-rpath,./cuda -o $@

crossmatch.o: reducer.cpp crossmatch.h
	$(CXX) $(CXXFLAGS) $< -o $@

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
	rm -f *.o $(EXECUTABLE)
