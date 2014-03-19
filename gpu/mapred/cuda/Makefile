TARGET = libgpuspatial.so
CC= nvcc

all: $(TARGET)

$(TARGET): pixelgrid.cu cuda_spatial.h
	${CC} -arch=sm_20 -O --ptxas-options=-v -Xcompiler -fPIC -shared $< -o $@
	cp $(TARGET) ~/softs/lib/

clean:
	rm -f *.o $(TARGET)

