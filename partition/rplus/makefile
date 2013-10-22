
FLAGS= -O2

SRC= main.c index.c rectangl.c printsta.c clock.c allocate.c error.c\
        pack.c insert.c splitnod.c partitio.c node.c page.c func.c

HEADERS= options.h macros.h index.h assert.h  partitio.h global.h

ALL= $(SRC) $(HEADERS)

OBJ= main.o index.o rectangl.o printsta.o clock.o allocate.o error.o\
        pack.o insert.o splitnod.o partitio.o node.o page.o func.o

all: rpt

#.c.o: $(HEADERS)
$(OBJ): $(HEADERS) $(SRC)
	gcc -m64 -c $(FLAGS) $*.c

rpt: $(OBJ)
	gcc -m64 $(FLAGS) $(OBJ) -lm -o $@

clean:
	rm -f *.o rpt
