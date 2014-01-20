spatial=/home/aaji/softs

UNAME		= $(shell uname)
OPTFLAGS	= -O2
INCBASE		= -I. -I../../
LIBBASE		= -lspatialindex -lboost_program_options
# -lboost_program_options

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
    #CC = clang
    #CXX = clang++
else 
    CC = gcc
    CXX = g++
endif

TARGET = serialhc

all: $(TARGET)

debug: CXX += -DDEBUG -g
debug: CC += -DDEBUG -g
debug: all 

serialhc: serialhc.cc
	$(CXX) $< $(CFLAGS) $(LDFLAGS) -o $@

clean:
	@rm -f $(TARGET)

