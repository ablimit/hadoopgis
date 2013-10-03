#! /bin/bash

echo "++++++++++++++++++++pais+++++++++++++++++++++"
./mapper  -d pais -w 0 -s 0 -n 57344 -e 110592 -x 216 -y 112 -p pais < pais.test.wkt > pais.out.txt

echo ""

#echo "++++++++++++++++++++osm+++++++++++++++++++++"
#./mapper  -d osm -w -180 -s -90 -n 90 -e 180 -x 4 -y 4 -p osm < osm.test.wkt  > osm.out.txt 


