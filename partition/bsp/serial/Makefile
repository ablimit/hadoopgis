spatial=/home/aaji/softs
hoangboost=/home/hoang/Downloads/boost_1_48_0

UNAME		= $(shell uname)
OPTFLAGS	= -O2 -Wall -std=c++0x
INCBASE		= -I. -I../../ -I${hoangboost}
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

CFLAGS      = $(OPTFLAGS) $(INCBASE) $(INCFLAGS) 
LDFLAGS     = $(LIBBASE) $(LIBS)

ifeq ($(UNAME), Darwin)
    CC = gcc -arch x86_64
    CXX = g++ -arch x86_64
else 
    CC = gcc 
    CXX = g++
endif


TARGET = bsp
TARGET2 = reducerBsp
Obj = $(TARGET).o 

all: $(TARGET) ${TARGET2}

debug: CXX += -DDEBUG -g
debug: CC += -DDEBUG -g
debug: all 

reducerBsp: reducerBsp.cpp commonspatial.h $(Obj)
	$(CXX) $^ $(CFLAGS) $(LDFLAGS) -o $@

bsp: main.cpp commonspatial.h $(Obj)
	$(CXX) $^ $(CFLAGS) $(LDFLAGS) -o $@

bsp.o: BinarySplitNode.cpp commonspatial.h
	$(CXX) -c $< $(CFLAGS) $(LDFLAGS) -o $@

clean:
	@rm -f $(TARGET) $(Obj)
