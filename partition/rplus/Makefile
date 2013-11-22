boost=/home/aaji/softs/boost
spatial=/home/aaji/softs

UNAME		= $(shell uname)
OPTFLAGS	= -O2
INCBASE		= -I.
LIBBASE		= -lspatialindex -lgeos

ifneq (,$(findstring emory,$(shell uname -a)))
    INCFLAGS = -I$(spatial)/include
    LIBS = -L$(spatial)/lib
else 
    ifeq ($(UNAME), Darwin)
    INCFLAGS = -I /Library/Frameworks/GEOS.framework/Headers -I /usr/local/include
    LIBS = -L /Library/Frameworks/GEOS.framework/unix/lib -L /usr/local/lib
endif
endif

CFLAGS      = $(OPTFLAGS) $(INCBASE) $(INCFLAGS) 
LDFLAGS     = $(LIBBASE) $(LIBS)

ifeq ($(UNAME), Darwin)
    CC = g++ -arch x86_64
else 
    CC = g++
endif

all: genRplusPartition genRtreeIndex rquery Makefile

debug: CXX += -DDEBUG -g
debug: CC += -DDEBUG -g
debug: all 


genRplusPartition: RplusPartitioner.cc SpaceStreamReader.h
	$(CC) RplusPartitioner.cc -std=c++0x $(CFLAGS) $(LDFLAGS) -o genRplusPartition

#genRplusIndex: RplusBulkLoad.cc 
#	$(CC) RplusBulkLoad.cc $(CFLAGS) $(LDFLAGS) -o genRplusIndex
	
genRtreeIndex: RTreeBulkLoad.cc
	$(CC) RTreeBulkLoad.cc $(CFLAGS) $(LDFLAGS) -o genRtreeIndex

#genPartitionRegionFromIndex: RplusPartitionMBB.cc dbg.h
#	$(CC) RplusPartitionMBB.cc $(CFLAGS) $(LDFLAGS) -o genPartitionRegionFromIndex

rquery: RTreeQuery.cc
	$(CC) RTreeQuery.cc $(CFLAGS) $(LDFLAGS) -o rquery


clean:
	@rm -f genRtreeIndex genPartitionRegionFromIndex rquery genRplusIndex genRplusPartition


