
dir=/home/aaji/softs/boost
spatial=/home/aaji/softs

all: mapper
    
mapper: mapper1.cpp mapper2.cpp mapper3.cpp mapper4.cpp hadoopgis.h
	g++ -m64 -O3 -lspatialindex mapper1.cpp -o mapper1 -I${dir}/include -L${dir}/lib -I${spatial}/include -L${spatial}/lib
	g++ -m64 -O3 -lspatialindex mapper2.cpp -o mapper2 -I${dir}/include -L${dir}/lib -I${spatial}/include -L${spatial}/lib
	g++ -m64 -O3 -lspatialindex mapper3.cpp -o mapper3 -I${dir}/include -L${dir}/lib -I${spatial}/include -L${spatial}/lib
	g++ -m64 -O3 -lspatialindex mapper4.cpp -o mapper4 -I${dir}/include -L${dir}/lib -I${spatial}/include -L${spatial}/lib
clean:
	rm mapper1
	rm mapper2
	rm mapper3
	rm mapper4

