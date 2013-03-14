
boost=/home/aaji/softs/boost
spatial=/home/aaji/softs

UNAME		= $(shell uname)
OPTFLAGS	= -O3
INCBASE		= -I.
LIBBASE		= 

ifneq (,$(findstring emory,$(shell uname -a)))
	INCFLAGS = -I$(boost)/include
	LIBS = -L$(boost)/lib
endif

CFLAGS      = $(OPTFLAGS) $(INCBASE) $(INCFLAGS) 
LDFLAGS     = $(LIBBASE) $(LIBS)

ifeq ($(UNAME), Darwin)
	CC = g++ -arch x86_64
else 
	CC = g++
endif

all: regen mapa reda redb

regen: regenerator.cpp
	$(CC) regenerator.cpp $(CFLAGS) $(LDFLAGS) -o regen

mapa: mapperStarDup2.cpp
	$(CC) mapperStarDup2.cpp $(CFLAGS) $(LDFLAGS) -o mappera

reda: reducerStarDup2.cpp
	$(CC) reducerStarDup2.cpp $(CFLAGS) $(LDFLAGS) -o reducera

redb: reducerRemove2.cpp
	$(CC) reducerRemove2.cpp $(CFLAGS) $(LDFLAGS) -o reducerb

clean:
	@rm -f regen

