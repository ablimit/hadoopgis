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
    INCFLAGS = -I /Library/Frameworks/GEOS.framework/Headers
    LIBS = -L /Library/Frameworks/GEOS.framework/unix/lib
endif
endif

CFLAGS      = $(OPTFLAGS) $(INCBASE) $(INCFLAGS) 
LDFLAGS     = $(LIBBASE) $(LIBS)

ifeq ($(UNAME), Darwin)
    CC = g++ -arch x86_64
else 
    CC = g++
endif

objetcs = loader parmbb pquery

all: $(objetcs)

loader: RTreeBulkLoadOSM.cc
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
	
parmbb: RTreePartitionMBB.cc
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@

pquery: RTreeQuery.cc
	$(CC) $^  $(CFLAGS) $(LDFLAGS) -o $@

clean :
	rm $(objects)

