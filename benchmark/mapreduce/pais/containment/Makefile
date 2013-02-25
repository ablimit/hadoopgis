
boost=/home/aaji/softs/boost
spatial=/home/aaji/softs

UNAME 		= $(shell uname)
OPTFLAGS    = -O3
INCFLAGS    = -I.
LIBS        = -lspatialindex 

ifneq (,$(findstring emory,$(shell uname -a)))
	INCFLAGS = $(INCFLAGS) -I$(boost)/include -I$(spatial)/include
	LIBS = $(LIBS) -L$(boost)/lib -L$(spatial)/lib
endif

CFLAGS      = $(OPTFLAGS) $(INCFLAGS) 

ifeq ($(UNAME), Darwin)
	CC = g++ -arch x86_64
else 
	CC = g++
endif

all: q1 q2 q3 q4

q1: query1.cpp hadoopgis.h vecstream.h
	$(CC) query1.cpp $(CFLAGS) $(LIBS) -o q1

q2: query2.cpp hadoopgis.h vecstream.h
	$(CC) query2.cpp $(CFLAGS) $(LIBS) -o q2

q3: query3.cpp hadoopgis.h vecstream.h
	$(CC) query3.cpp $(CFLAGS) $(LIBS) -o q3

q4: query4.cpp hadoopgis.h vecstream.h
	$(CC) query4.cpp $(CFLAGS) $(LIBS) -o q4

clean:
	@rm -f q1 q2 q3 q4

