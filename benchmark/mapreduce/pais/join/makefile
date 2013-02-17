dir=/home/aaji/softs/boost
spatial=/home/aaji/softs

all: mapper reducer
    
mapper:	mapper.cpp
	g++ -m64 -O3 mapper.cpp -o mapper -I${dir}/include -L${dir}/lib

reducer: reducerx.cpp IndexParam.h MReducerx.h
	g++ -m64 -O3 -lspatialindex reducerx.cpp -o reducer -I${dir}/include -L${dir}/lib -I${spatial}/include -L${spatial}/lib
clean:
	rm mapper
	rm reducer

