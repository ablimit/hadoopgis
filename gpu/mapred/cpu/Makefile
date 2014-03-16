
TARGET = libcpuspatial.so

CC= g++
CFLAGS= -O2 -Wall -fPIC -shared -Wl,-soname,$(TARGET).1
LDFLAGS = -L/usr/local/lib

all: $(TARGET)

$(TARGET): pixelgrid.cpp cpu_spatial.h
	${CC} ${CFLAGS} ${LDFLAGS} $< -o $@.1.0.1

clean:
	rm -f *.o $(TARGET)


