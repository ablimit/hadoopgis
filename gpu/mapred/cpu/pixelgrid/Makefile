TARGET = libcpuspatial.so

CC= g++
CFLAGS= -O2 -Wall -fPIC -shared -Wl,-soname,$(TARGET)
LDFLAGS = -L/usr/local/lib

all: $(TARGET)

debug: CXX += -pg
debug: CC += -pg
debug: all


$(TARGET): pixelgrid.cpp
	${CC} ${CFLAGS} ${LDFLAGS} $< -o $@

install: 
	cp $(TARGET) ~/softs/lib/

clean:
	rm -f *.o $(TARGET)

