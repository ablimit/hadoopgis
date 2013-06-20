#! /bin/bash

# file path
path=/data2/ablimit/Data/spatialdata/osmout
outpath=/scratch/aaji/osm
f1=${path}/planet.1000x1000.dat.1
f2=${path}/europe.1000x1000.dat.2

if [ ! -e genosmmbb ] ; then 
    echo "MBR calculator [genosmmbb] is missing."
    exit 0;
fi

if [ ! -e genpid ] ; then 
    echo "oid -- > pid Mapper is missing. [genpid]"
    exit 0;
fi

if [ ! -e ${outpath} ] ;
then
    echo "Path ${outpath} does not exist. Creating it..."
    mkdir -p ${outpath}
fi

echo "Generating MBRs for planet"
#echo "genosmmbb 1 < ${f1} | python normalize.py osm | gzip > ${outpath}/mbb.planet.txt.gz "

echo "Generating MBRs for europe"
#echo "genosmmbb 2 < ${f2} | python normalize.py osm | gzip > ${outpath}/mbb.europe.txt.gz"
# exit 0;


# echo "Re-Partition the map"
for size in 50000 100000 200000 300000 400000 500000
do
    mark=${size%000}
    dir=partres/osm/oc${mark}k
    outdir=${outpath}/partition/oc${mark}k

    echo "processing set oc${mark}k"
    mkdir -p ${outdir}

    for method in rtree minskew minskewrefine rv rkHist sthist 
    do
	if [ -e ${dir}/osm.${method}.txt ] && [ ! -e ${outdir}/osm.${method}.txt.gz ] ; then

	    echo "resharding for method ${method} .."

	    zcat ${outpath}/mbb.planet.txt.gz  ${outpath}/mbb.europe.txt.gz | genpid ${dir}/osm.${method}.txt | python reshardosm.py ${f1} ${f2} | gzip > ${outdir}/osm.${method}.txt.gz 
	fi
    done
done

