#! /bin/bash

# file path
ipath=/data2/ablimit/Data/spatialdata/osmout
# opath=/mnt/scratch1/aaji/fixedgrid/osm
opath=/scratch/aaji/partition/osm/fixedgrid
f1=${ipath}/planet.1000x1000.dat.1
f2=${ipath}/europe.1000x1000.dat.2
f3=${ipath}/osm.mbb.norm.filter.dat

for size in 2000 2500 3000 3500 4000 8000 10000 50000 100000 200000 300000 500000
do

    if [ ! -e ${opath}/grid${size} ] ;
    then 
	mkdir -p ${opath}/grid${size}
    fi

    echo "Grid size: ${size}x${size}"

    echo "FixedGrid for $f3: "
    START=$(date +%s)
    ./partmbb  -d osm -w 0 -s 0 -n 1.0 -e 1.0 -x ${size} -y ${size}  < ${f3} > ${opath}/grid${size}/part.txt
    END=$(date +%s)
    DIFF=$(( $END - $START ))
    echo "Time: ${DIFF} secs."

   # legacy code
   # echo "FixedGrid for $f2: "
   # START=$(date +%s)
   # ./mapper  -d osm -w -180 -s -90 -n 90 -e 180 -x ${size} -y ${size}  < ${f2} > ${opath}/grid${size}/europe.2
   # END=$(date +%s)
   # DIFF=$(( $END - $START ))
   # echo "Time: ${DIFF} secs."
done

