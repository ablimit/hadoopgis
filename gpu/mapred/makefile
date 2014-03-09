CC= gcc
CFLAGS= -c -O2 -Wall
LDFLAGS= -L./cuda -lgpuspatial
SOURCES= reducer.c spatialindex.c spatial.c

EXECUTABLE = cudareducer
OBJECTS = spatial.o spatialindex.o

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) crossmatch.h
	$(CC) $(LDFLAGS) $(OBJECTS) -Wl,-rpath,./cuda -o $@

spatial.o: spatial.c spatial.h
	$(CC) $(CFLAGS) $< -o $@

spatialindex.o: spatialindex.c spatialindex.h
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o $(EXECUTABLE)
