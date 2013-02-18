
dir=/home/aaji/softs/boost
spatial=/home/aaji/softs

all: mapper
    
mapper: mapper.cpp
	g++ -m64 -O3 -lspatialindex mapper.cpp -o mapper # -I${dir}/include -L${dir}/lib -I${spatial}/include -L${spatial}/lib
clean:
	rm mapper

