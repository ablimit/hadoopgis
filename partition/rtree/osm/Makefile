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

# all: process loader plot part mbb view
all: loader mbb query

process: pais/GeneratePAISMBB.cc osm/GenerateOSMMBB.cc
	#$(CC) pais/GeneratePAISMBB.cc $(CFLAGS) $(LDFLAGS) -o pais/genpaismbb
	$(CC) osm/GenerateOSMMBB.cc $(CFLAGS) $(LDFLAGS) -o osm/genosmmbb

loader: RTreeBulkLoadOSM.cc RTreeBulkLoadPAIS.cc
	$(CC) RTreeBulkLoadOSM.cc $(CFLAGS) $(LDFLAGS) -o genRtreeIndexOsm
	$(CC) RTreeBulkLoadPAIS.cc $(CFLAGS) $(LDFLAGS) -o genRtreeIndexPais
	
plot: genPlotFromIndex.cc
	$(CC) genPlotFromIndex.cc $(CFLAGS) $(LDFLAGS) -o genPlotFromIndex

part: RTreePartitionFromIndex.cc
	$(CC) RTreePartitionFromIndex.cc $(CFLAGS) $(LDFLAGS) -o genPartitionFromIndex

mbb: RTreePartitionMBB.cc
	$(CC) RTreePartitionMBB.cc $(CFLAGS) $(LDFLAGS) -o genPartitionRegionFromIndex

query: RTreeQuery.cc
	$(CC) RTreeQuery.cc $(CFLAGS) $(LDFLAGS) -o rquery

view: RTreeView.cc
	$(CC) RTreeView.cc $(CFLAGS) $(LDFLAGS) -o viewRTree

genpid: oidmatchpid.cc
	$(CC) oidmatchpid.cc $(CFLAGS) $(LDFLAGS) -o genpid

clean:
	@rm -f genRtreeIndexOsm genRtreeIndexPais genPlotFromIndex genPartitionFromIndex genPartitionRegionFromIndex viewRTree rquery
	@rm -f pais/genpaismbb osm/genosmmbb
	@rm -f genpid


