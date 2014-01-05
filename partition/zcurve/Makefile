spatial=/home/aaji/softs

UNAME		= $(shell uname)
OPTFLAGS	= -O2
INCBASE		= -I.


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

TARGET = test_hilbert 

all: $(TARGET) test_ieee

test_hilbert: hilbert.c hilbert.h
	$(CC) -DTEST_HILBERT $< $(CFLAGS) $(LDFLAGS) -o $@

test_ieee: hilbert.c hilbert.h
	$(CC) -DTEST_IEEE $< $(CFLAGS) $(LDFLAGS) -o $@

clean:
	@rm -f $(TARGET)

