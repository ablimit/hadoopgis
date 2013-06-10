#! /bin/bash

# file path
path=/data2/ablimit/Data/spatialdata/osmout
outpath=/scratch/aaji/osm
f1=${path}/planet.1000x1000.dat.1
f2=${path}/europe.1000x1000.dat.2

if [ ! -e genosmmbb ] ;
then 
    echo "MBR calculator [genosmmbb] is missing."
    exit 0;
fi

if [ ! -e ${outpath} ] ;
then
    echo "Path ${outpath} does not exist. Creating it..."
    mkdir -p ${outpath}
fi

echo "Generating MBRs for planet"
genosmmbb 1 < ${f1} | python normalize.py osm | bzip2 > ${outpath}/mbb.planet.txt.bz2 

echo "Generating MBRs for europe"
genosmmbb 2 < ${f2} | python normalize.py osm | bzip2 > ${outpath}/mbb.europe.txt.bz2

mark=1201

echo "Re-Partition the map"
for method in rtree minskew minskewrefine rv rkHist sthist 
do
    if [ -e partres/osm/osm.${mark}.${method}.txt ]
    then
	echo "resharding for method ${method} .."
	outdir=${outpath}/partition/part${method}
	if [ -e ${outdir} ]
	then
	    mkdir -p ${outdir}
	fi

	bzcat ${outpath}/mbb.planet.txt.bz2  ${outpath}/mbb.europe.txt.bz2 | python genpid.py partres/osm/osm.${mark}.${method}.txt | python reshardosm.py ${f1} ${f2} | bzip2 > ${outdir}/osm.txt.bz2 
    fi
done

