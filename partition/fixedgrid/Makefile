
boost=/home/aaji/softs/boost
spatial=/home/aaji/softs

UNAME		= $(shell uname)
OPTFLAGS	= -O2
INCBASE		= -I.
LIBBASE		= -lspatialindex -lgeos

ifneq (,$(findstring emory,$(shell uname -a)))
    INCFLAGS = -I$(boost)/include -I$(spatial)/include
    LIBS = -L$(boost)/lib -L$(spatial)/lib
else 
    ifeq ($(UNAME), Darwin)
    INCFLAGS = -I /Library/Frameworks/GEOS.framework/Headers
    LIBS = -L /Library/Frameworks/GEOS.framework/unix/lib
endif
endif

CFLAGS      = $(OPTFLAGS) $(INCBASE) $(INCFLAGS) 
LDFLAGS     = $(LIBBASE) $(LIBS)

ifeq ($(UNAME), Darwin)
    CC = g++ -arch x86_64
else 
    CC = g++
endif

all: part

cmdline_mapper.o: options_mapper.ggo
	gengetopt -i options_mapper.ggo -F cmdline_mapper
	$(CC) -c cmdline_mapper.c -o cmdline_mapper.o

cmdline_reducer.o: options_reducer.ggo
	gengetopt -i options_reducer.ggo -F cmdline_reducer
	$(CC) -c cmdline_reducer.c -o cmdline_reducer.o

map: cmdline_mapper.o mapper.cpp hadoopgis.h
	$(CC) mapper.cpp cmdline_mapper.o $(CFLAGS) $(LDFLAGS) -o mapper

part: cmdline_mapper.o partitionmbb.cpp hadoopgis.h
	$(CC) partitionmbb.cpp cmdline_mapper.o $(CFLAGS) $(LDFLAGS) -o partmbb

reduce: cmdline_reducer.o reducer.cpp hadoopgis.h
	$(CC) reducer.cpp cmdline_reducer.o $(CFLAGS) $(LDFLAGS) -o reducer

clean:
	@rm -f mapper reducer
	@rm -f cmdline*
	@rm -f *.o

