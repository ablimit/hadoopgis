
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

all: regen mapa reda redb redb2 mapb2

regen: regenerator.cpp
	$(CC) regenerator.cpp $(CFLAGS) $(LDFLAGS) -o regen

mapa: mapperStarDup2.cpp
	$(CC) mapperStarDup2.cpp $(CFLAGS) $(LDFLAGS) -o mappera

reda: reducerStarDup2.cpp
	$(CC) reducerStarDup2.cpp $(CFLAGS) $(LDFLAGS) -o reducera

redb: reducerRemove2.cpp
	$(CC) reducerRemove2.cpp $(CFLAGS) $(LDFLAGS) -o reducerb

mapb2: mapperRemove3.cpp
	$(CC) mapperRemove3.cpp $(CFLAGS) $(LDFLAGS) -o mapperb2

redb2: reducerRemove3.cpp
	$(CC) reducerRemove3.cpp $(CFLAGS) $(LDFLAGS) -o reducerb2

identity: mapperIdentity.cpp
	$(CC) mapperIdentity.cpp -o mapperidentity

clean:
	@rm -f regen mapa reda redb mapb2 redb2 mapperidentity

