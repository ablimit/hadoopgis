UNAME		= $(shell uname)
OPTFLAGS	= -O2
INCBASE		= -I.
LIBBASE		= 

ifneq (,$(findstring emory,$(shell uname -a)))
    # INCFLAGS = -I$(boost)/include -I$(spatial)/include
    #LIBS = -L$(boost)/lib -L$(spatial)/lib
else 
    ifeq ($(UNAME), Darwin)
    #INCFLAGS = -I /Library/Frameworks/GEOS.framework/Headers
    #LIBS = -L /Library/Frameworks/GEOS.framework/unix/lib
endif
endif

CFLAGS      = $(OPTFLAGS) $(INCBASE) $(INCFLAGS) 
LDFLAGS     = $(LIBBASE) $(LIBS)

ifeq ($(UNAME), Darwin)
    CC = gcc -arch x86_64
else 
    CC = gcc
endif

all: compile

compile: polyclip.c node.h
	$(CC) polyclip.c $(CFLAGS) $(LDFLAGS) -o clip

clean:
	@rm -f clip
