spatial=/home/aaji/softs

UNAME		= $(shell uname)
OPTFLAGS	= -O2
INCBASE		= -I. -I../../
LIBBASE		= -lspatialindex -lboost_program_options

ifneq (,$(findstring emory,$(shell uname -a)))
    INCFLAGS = -I$(spatial)/include
    LIBS = -L$(spatial)/lib
else 
    ifeq ($(UNAME), Darwin)
    INCFLAGS = -I /usr/local/include
    LIBS = -L /usr/local/lib
endif
endif

CFLAGS      = -std=c++0x $(OPTFLAGS) $(INCBASE) $(INCFLAGS) 
LDFLAGS     = $(LIBBASE) $(LIBS)

ifeq ($(UNAME), Darwin)
    CC = gcc -arch x86_64
    CXX = g++ -arch x86_64
else 
    CC = gcc
    CXX = g++
endif

TARGET = fg bsp hc slc bos str

all: $(TARGET)

debug: CXX += -DDEBUG -g
debug: CC += -DDEBUG -g
debug: all 

fg: serialfg.cc
	$(CXX) $< $(CFLAGS) $(LDFLAGS) -o $@

bsp.o: BinarySplitNode.cpp commonspatial.h
	$(CXX) -c $< $(CFLAGS) $(LDFLAGS) -o $@

bsp: main.cpp commonspatial.h bsp.o
	$(CXX) $^ $(CFLAGS) $(LDFLAGS) -o $@

hc: serialhc.cc
	$(CXX) $< $(CFLAGS) $(LDFLAGS) -o $@

slc: serialslc.cc
	$(CXX) $< $(CFLAGS) $(LDFLAGS) -o $@

bos: serialbos.cc
	$(CXX) $< $(CFLAGS) $(LDFLAGS) -o $@

str: serialstr.cc
	$(CXX) $< $(CFLAGS) $(LDFLAGS) -o $@

clean:
	@rm -f $(TARGET) *.o

