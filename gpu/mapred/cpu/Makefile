TARGET = libcpuspatial.so

CC= gcc
CFLAGS= -O2 -Wall -fPIC -shared -Wl,-soname,$(TARGET)
LDFLAGS = -L/usr/local/lib

all: $(TARGET)

debug: CXX += -pg
debug: CC += -pg
debug: all


$(TARGET): pixelgrid.cpp cpu_spatial.h
	${CC} ${CFLAGS} ${LDFLAGS} $< -o $@
	cp $(TARGET) ~/softs/lib/

clean:
	rm -f *.o $(TARGET)

