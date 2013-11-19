boost=/home/aaji/softs/boost
spatial=/home/aaji/softs

UNAME		= $(shell uname)
OPTFLAGS	= -O2
INCBASE		= -I.
LIBBASE		= -lspatialindex -lgeos

ifneq (,$(findstring emory,$(shell uname -a)))
    INCFLAGS = -I$(boost)/include -I$(spatial)/include
    LIBS = -L$(boost)/lib -L$(spatial)/lib
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

# all: process loader plot part mbb view
all: genRtreeIndex genPartitionRegionFromIndex rquery

genRtreeIndex: RTreeBulkLoadOSM.cc
	$(CC) RTreeBulkLoadOSM.cc $(CFLAGS) $(LDFLAGS) -o genRtreeIndex
	
genPartitionRegionFromIndex: RTreePartitionMBB.cc
	$(CC) RTreePartitionMBB.cc $(CFLAGS) $(LDFLAGS) -o genPartitionRegionFromIndex

rquery: RTreeQuery.cc
	$(CC) RTreeQuery.cc $(CFLAGS) $(LDFLAGS) -o rquery


clean:
	@rm -f genRtreeIndex genPartitionRegionFromIndex rquery


