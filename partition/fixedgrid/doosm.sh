#! /bin/bash

# file path
ipath=/data2/ablimit/Data/spatialdata/osmout
# opath=/mnt/scratch1/aaji/fixedgrid/osm
opath=/scratch/aaji/partition/fixedgrid/osm
f1=${ipath}/planet.1000x1000.dat.1
f2=${ipath}/europe.1000x1000.dat.2

# for size in 2000 2500 3000 3500 4000 8000
for size in 8000
do

    if [ ! -e ${opath}/grid${size} ] ;
    then 
	mkdir -p ${opath}/grid${size}
    fi

    echo "Grid size: ${size}x${size}"

    echo "FixedGrid for $f1: "
    START=$(date +%s)
    ./mapper  -d osm -w -180 -s -90 -n 90 -e 180 -x ${size} -y ${size}  < ${f1} > ${opath}/grid${size}/planet.1
    END=$(date +%s)
    DIFF=$(( $END - $START ))
    echo "Time: ${DIFF} secs."

    echo "FixedGrid for $f2: "
    START=$(date +%s)
    ./mapper  -d osm -w -180 -s -90 -n 90 -e 180 -x ${size} -y ${size}  < ${f2} > ${opath}/grid${size}/europe.2
    END=$(date +%s)
    DIFF=$(( $END - $START ))
    echo "Time: ${DIFF} secs."
done

