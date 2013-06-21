
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

all: map reduce fgmap

map: mapper.cpp
	$(CC) mapper.cpp $(CFLAGS) $(LDFLAGS) -o mapper

reduce: reducer.cpp MReducerx.h
	$(CC) reducer.cpp $(CFLAGS) $(LDFLAGS) -o reducer

fgmap: fgmapper.cpp
	$(CC) fgmapper.cpp $(CFLAGS) $(LDFLAGS) -o fgmapper

clean:
	@rm -f mapper reducer

