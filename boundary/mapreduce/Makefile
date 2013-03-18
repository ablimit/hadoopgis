
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

all: map red

map: mapper.cpp
	$(CC) mapper.cpp $(CFLAGS) $(LDFLAGS) -o mapper

red: reducer.cpp
	$(CC) reducer.cpp $(CFLAGS) $(LDFLAGS) -o reducer

clean:
	@rm -f mapper reducer

