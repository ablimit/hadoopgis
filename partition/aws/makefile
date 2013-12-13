
prefix=/usr/local

libbase=/usr/local/lib
incbase=/usr/local/include

OPTFLAGS	= -O2
INCFLAGS	= -I $(incbase)
LIBS		= -lgeos -lspatialindex -L $(libbase)

CC = gcc
CXX = g++


all: resque

resque: effresque.cpp
	$(CXX) $< $(INCFLAGS) $(LIBS) $(OPTFLAGS) -o $@

install:
	mkdir -p $(prefix)/bin
	cp resque $(prefix)/bin

clean:
	@rm -f resque


