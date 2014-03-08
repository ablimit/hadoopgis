CC= gcc
CFLAGS= -c -O2 -Wall
LDFLAGS= -L./cuda -lcudaspatial
SOURCES= main.c justdoit.c spatialindex.c rtree.c spatial.c hilbert.c rstar.c
#SOURCES= testcuda.c justdoit.c spatialindex.c rtree.c spatial.c hilbert.c rstar.c
OBJECTS= $(SOURCES:.cpp=.o)
EXECUTABLE= spatialjoin

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -Wl,-rpath,./cuda -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o $(EXECUTABLE)
