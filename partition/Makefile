spatial=/home/aaji/softs

UNAME		= $(shell uname)
OPTFLAGS	= -O2
INCBASE		= -I.
LIBBASE		= -lspatialindex

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
    # CC = clang
    # CXX = clang++
else 
    CC = gcc
    CXX = g++
endif

TARGET = fixedgridPartition stripGroupPartition rplusGroupPartition hilbertPartition genRtreeIndex genPartitionFromIndex rquery

all: $(TARGET)

debug: CXX += -DDEBUG -g
debug: CC += -DDEBUG -g
debug: all 

fixedgridPartition: FixedGridPartitioner.cc SpaceStreamReader.h
	$(CXX) $< $(CFLAGS) $(LDFLAGS) -o $@

genRplusPartition: RplusPartitioner.cc SpaceStreamReader.h
	$(CXX) $< -std=c++0x $(CFLAGS) $(LDFLAGS) -o $@

rplusGroupPartition: RplusGroupPartitioner.cc SpaceStreamReader.h
	$(CXX) $< -std=c++0x $(CFLAGS) $(LDFLAGS) -o $@

stripGroupPartition: StripGroupPartitioner.cc SpaceStreamReader.h
	$(CXX) $< -std=c++0x $(CFLAGS) $(LDFLAGS) -o $@

hilbertPartition: HilbertPartitioner.cc SpaceStreamReader.h
	$(CXX) $< -std=c++0x $(CFLAGS) $(LDFLAGS) -o $@

genRtreeIndex: RTreeBulkLoad.cc
	$(CXX) $< $(CFLAGS) $(LDFLAGS) -o $@

genPartitionFromIndex: RTreePartitionMBB.cc
	$(CXX) $< $(CFLAGS) $(LDFLAGS) -o $@

rquery: RTreeQuery.cc
	$(CXX) $< $(CFLAGS) $(LDFLAGS) -o $@

clean:
	@rm -f $(TARGET)

