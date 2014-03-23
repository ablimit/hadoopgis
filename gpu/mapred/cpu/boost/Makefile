TARGET = libcpuspatial.so

CC= g++
CFLAGS= -O2 -Wall -fPIC -shared -Wl,-soname,$(TARGET)
LDFLAGS = -L/usr/local/lib

all: $(TARGET)

debug: CXX += -pg
debug: CC += -pg
debug: all


$(TARGET): polyclip.cpp
	${CC} ${CFLAGS} ${LDFLAGS} $< -o $@

clean:
	rm -f *.o $(TARGET)

install:
	cp $(TARGET) ~/softs/lib/
