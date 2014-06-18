SHARED_TARGET = libcpurefine.so
STATIC_TARGET = libcpurefine.a
TARGET_OBJ = libcpurefine.o

CC= g++
CFLAGS= -O2 -m64 -Wall 
LDFLAGS = -L/usr/local/lib
INCFLAGS = -I.. -I../..

all: $(SHARED_TARGET) $(STATIC_TARGET)

debug: CXX += -pg
debug: CC += -pg
debug: all

$(SHARED_TARGET): polyclip.cpp ../cpu_refine.h
	${CC} ${CFLAGS} -fPIC -shared -Wl,-soname,$(SHARED_TARGET) ${INCFLAGS} ${LDFLAGS} $< -o $@

$(STATIC_TARGET): $(TARGET_OBJ)
	ar rvs $@ $<

$(TARGET_OBJ): polyclip.cpp ../cpu_refine.h
	${CC} ${CFLAGS} -c ${INCFLAGS} ${LDFLAGS} $< -o $@

clean:
	rm -f *.o *.a *.so

install:
	cp $(SHARED_TARGET) ~/softs/lib/
	cp $(STATIC_TARGET) ~/softs/lib/
