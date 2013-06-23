#! /bin/bash
if [ $# -lt 1 ]; then
    echo "Usage: $0 [partition size=100000]"
    exit 0
fi

# jvm params 
jvm="-Xss4m -Xmx100G"

# file path
path=/data2/ablimit/Data/spatialdata/osmout
osmmbb=/data2/ablimit/Data/spatialdata/sigspatial2013/osm.mbb.filter.txt
f1=${path}/planet.1000x1000.dat.1
f2=${path}/europe.1000x1000.dat.2
temp=/scratch/aaji/temp/
outpath=/scratch/aaji/osm
# partition parameters
gridsize=11
alpha=0.1
ratio=0.4
sample=0.4

if [ ! -e genosmmbb ] ;
then 
    echo "MBR calculator [genosmmbb] is missing."
    exit 0;
fi

if [ ! -e ${temp} ] ;
then
    echo "Temp ${temp} does not exist. Creating it..."
    mkdir -p ${temp}
fi


# echo "calculating MBRs for dataset."
# co-partition the dataset 
echo "MBB+Norm -- planet"
# cat ${f1}  | genosmmbb 1 | python data/normalize.py osm | gzip > ${outpath}/mbb.planet.txt.gz
echo "MBB+Norm -- europe"
# cat ${f2}  | genosmmbb 2 | python data/normalize.py osm | gzip > ${outpath}/mbb.europe.txt.gz
echo "Filter"
zcat ${outpath}/mbb.planet.txt.gz ${outpath}/mbb.europe.txt.gz | python data/filterosm.py | gzip >  ${outpath}/osm.mbb.filter.txt.gz
exit 0 ;

# lc=`wc -l ${outpath}/osm.mbb.filter.txt | cut -d' ' -f1 `
lc=120167664

size=$1
# echo "size $size"
# exit 0;

#for size in 50000 100000 200000 300000 400000 500000
#for size in 100000 200000 300000 400000 500000
#do
rm -f /scratch/aaji/temp/*
echo "calculating the partition size."
p=`expr $((lc/size))`

echo "partition size is: $p"

mark=${size%000}
dir=data/partres/osm/oc${mark}k

if [ ! -e ${dir} ] ;
then
    echo "Partition dir ${dir} does not exist. Creating it..."
    mkdir -p ${dir}
fi

java ${jvm} -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.TestH1 $p ${osmmbb} ${dir}/osm ${temp} ${gridsize} ${alpha} ${ratio} ${sample}

#done

