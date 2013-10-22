# @(#)Makefile	1.13 3/13/84

FLAGS= -g3 -O

SRC= main.c index.c rectangl.c printsta.c clock.c allocate.c error.c\
        pack.c insert.c splitnod.c partitio.c node.c page.c func.c

HEADERS= options.h macros.h index.h assert.h  partitio.h global.h

ALL= $(SRC) $(HEADERS)

OBJ= main.o index.o rectangl.o printsta.o clock.o allocate.o error.o\
        pack.o insert.o splitnod.o partitio.o node.o page.o func.o


$(OBJ): $(HEADERS)

.c.o: $(HEADERS)
	cc -c $(FLAGS) $*.c

rpt: $(OBJ) 
	/bin/rm -f rpt
	cc $(FLAGS) $(OBJ) $(OBJLIN) -lm -o rpt


edit:
	vi $(SRC)
clean:
	rm -f *.o core tags

tags: $(SRC)
	ctags *.c

lint:
	rm -f lint.out
	lint *.c > lint.out

c:
	/bin/rm -f core
