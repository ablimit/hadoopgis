UNAME		= $(shell uname)
OPTFLAGS	= -O2
INCBASE		= -I. -I../../
LIBBASE		= -lspatialindex -lboost_program_options
INCFLAGS = -I /usr/local/include
LIBS = -L /usr/local/lib

CFLAGS      = $(OPTFLAGS) $(INCBASE) $(INCFLAGS) 
LDFLAGS     = $(LIBBASE) $(LIBS)

CC = gcc
CXX = g++

TARGET = slc

all: $(TARGET)

debug: CXX += -DDEBUG -g
debug: CC += -DDEBUG -g
debug: all 

slc: serialslc.cc
	$(CXX) $< -std=c++0x $(CFLAGS) $(LDFLAGS) -o $@

clean:
	@rm -f $(TARGET)

