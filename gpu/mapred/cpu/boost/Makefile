TARGET = libcpurefine.so

CC= g++
CFLAGS= -O2 -Wall -fPIC -shared -Wl,-soname,$(TARGET)
LDFLAGS = -L/usr/local/lib
INCFLAGS = -I../

all: $(TARGET)

debug: CXX += -pg
debug: CC += -pg
debug: all


$(TARGET): polyclip.cpp ../cpu_refine.h
	${CC} ${CFLAGS} ${INCFLAGS} ${LDFLAGS} $< -o $@

clean:
	rm -f *.o $(TARGET)

install:
	cp $(TARGET) ~/softs/lib/
