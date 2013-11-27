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

all: genRplusPartition rplusGroupPartition genRtreeIndex rquery Makefile

debug: CXX += -DDEBUG -g
debug: CC += -DDEBUG -g
debug: all 


genRplusPartition: RplusPartitioner.cc SpaceStreamReader.h
	$(CXX) RplusPartitioner.cc -std=c++0x $(CFLAGS) $(LDFLAGS) -o genRplusPartition

rplusGroupPartition: RplusGroupPartitioner.cc SpaceStreamReader.h
	$(CXX) RplusGroupPartitioner.cc -std=c++0x $(CFLAGS) $(LDFLAGS) -o rplusGroupPartition

#genRplusIndex: RplusBulkLoad.cc 
#	$(CC) RplusBulkLoad.cc $(CFLAGS) $(LDFLAGS) -o genRplusIndex
	
genRtreeIndex: RTreeBulkLoad.cc
	$(CXX) RTreeBulkLoad.cc $(CFLAGS) $(LDFLAGS) -o genRtreeIndex

#genPartitionRegionFromIndex: RplusPartitionMBB.cc dbg.h
#	$(CC) RplusPartitionMBB.cc $(CFLAGS) $(LDFLAGS) -o genPartitionRegionFromIndex

rquery: RTreeQuery.cc
	$(CXX) RTreeQuery.cc $(CFLAGS) $(LDFLAGS) -o rquery


clean:
	@rm -f genRtreeIndex genPartitionRegionFromIndex rquery genRplusIndex genRplusPartition rplusGroupPartition


