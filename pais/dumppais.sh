#! /bin/bash

echo "PAIS_UID|TILENAME|SEQUENCENUMBER|POLYGON" > /data2/ablimit/Data/spatialdata/pais/sixalgo.dat

for i in 1 2 3 4 5 6
do
    echo "dumping algo${i} ....."
    python dumphadoop2postcopy.py /data2/ablimit/algo${i} ${i} >> /data2/ablimit/Data/spatialdata/pais/sixalgo.dat
done

