CC= gcc
CXX= g++
CFLAGS= -c -O2 -Wall
LDFLAGS= -L./cuda -lgpuspatial
SOURCES= reducer.c spatialindex.c spatial.c

EXECUTABLE = cudareducer
OBJECTS = rtree.o hilbert.o rstar.o spatial.o spatialindex.o crossmatch.o

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -Wl,-rpath,./cuda -o $@

crossmatch.o: reducer.cpp crossmatch.h
	$(CXX) $(CFLAGS) $< -o $@

spatial.o: spatial.c spatial.h
	$(CC) $(CFLAGS) $< -o $@

spatialindex.o: spatialindex.c spatialindex.h
	$(CC) $(CFLAGS) $< -o $@

rtree.o: rtree.c rtree.h
	$(CC) $(CFLAGS) $< -o $@

rstar.o: rstar.c rstar.h
	$(CC) $(CFLAGS) $< -o $@

hilbert.o: hilbert.c hilbert.h
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o $(EXECUTABLE)
