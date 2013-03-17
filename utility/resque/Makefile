boost=/home/aaji/softs/boost
spatial=/home/aaji/softs

UNAME		= $(shell uname)
OPTFLAGS	= -O3
INCBASE		= -I.
LIBBASE		= -lspatialindex -lgeos

ifneq (,$(findstring emory,$(shell uname -a)))
	INCFLAGS = -I$(boost)/include -I$(spatial)/include
	LIBS = -L$(boost)/lib -L$(spatial)/lib
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

red: profilequery.cc MReducerx.h
	$(CC) profilequery.cc $(CFLAGS) $(LDFLAGS) -o profiler

clean:
	@rm -f mapper profiler

