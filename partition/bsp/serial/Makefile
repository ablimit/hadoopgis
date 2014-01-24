UNAME		= $(shell uname)
OPTFLAGS	= -O2
INCBASE		= -I.
LIBBASE		= -lboost_program_options
INCFLAGS = -I /usr/local/include
LIBS = -L /usr/local/lib

CFLAGS      = $(OPTFLAGS) $(INCBASE) $(INCFLAGS) 
LDFLAGS     = $(LIBBASE) $(LIBS)

CC = gcc
CXX = g++

TARGET = bsp
Obj = $(TARGET).o 

all: $(TARGET)

debug: CXX += -DDEBUG -g
debug: CC += -DDEBUG -g
debug: all 


bsp: main.cpp commonspatial.h $(Obj)
	$(CXX) $^ $(CFLAGS) $(LDFLAGS) -o $@

bsp.o: BinarySplitNode.cpp commonspatial.h
	$(CXX) -c $< $(CFLAGS) $(LDFLAGS) -o $@

clean:
	@rm -f $(TARGET) $(Obj)
