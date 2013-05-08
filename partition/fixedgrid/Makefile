
boost=/home/aaji/softs/boost
spatial=/home/aaji/softs

UNAME		= $(shell uname)
OPTFLAGS	= -O3
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

all: opts map

opts: options.ggo
	gengetopt -i options.ggo
	g++ -c cmdline.c -o cmdline.o

map: opts mapper.cpp hadoopgis.h  cmdline.h cmdline.c
	$(CC) mapper.cpp -c $(CFLAGS) $(LDFLAGS) -o mapper.o
	$(CC) $(CFLAGS) $(LDFLAGS) mapper.o cmdline.o -o mapper

reduce: opts reducer.cpp MReducerx.h
	$(CC) reducer.cpp $(CFLAGS) $(LDFLAGS) -o reducer

clean:
	@rm -f mapper reducer
	@rm -f reducer
	@rm -f cmdline.c cmdline.h
	@rm -f *.o

