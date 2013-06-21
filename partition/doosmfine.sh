#! /bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 [partition size=100000]"
    exit 0
fi

# jvm params 
jvm="-Xss4m -Xmx100G"

# file path
path=/data2/ablimit/Data/spatialdata
osmmbb=${path}/sigspatial2013/osm.mbb.filter.txt
f1=${path}/osmout/planet.1000x1000.dat.1
f2=${path}/osmout/europe.1000x1000.dat.2
temp=/scratch/aaji/temp/
outpath=/scratch/aaji/osm
# partition parameters
gridsize=8
alpha=0.1
ratio=0.4
sample=0.4

if [ ! -e ${temp} ] ;
then
    echo "Temp ${temp} does not exist. Creating it..."
    mkdir -p ${temp}
fi


# lc=`wc -l ${outpath}/osm.mbb.filter.txt | cut -d' ' -f1 `
lc=120167664

size=$1
# echo "size $size"
# exit 0;


for size in 50000 100000 200000 300000 400000 500000 
do
    mark=${size%000}
    dir=data/partres/osm/oc${mark}k

    for method in rtree minskew minskewrefine rv rkHist
    do
        if [ -e ${dir}/osm.${method}.txt ]; then

            # get mbb 
            subdir=data/partres/osm/oc${mark}k/sub
            while read pid x y xx yy area cc 
            do  
                python ${osmmbb} 
                for subsize in  500 1000 2000 3000 4000 5000
                do
                    echo "calculating the partition size."
                    p=`expr $((lc/size))`


                    echo "partition size is: $p"


                    if [ ! -e ${dir} ] ;
                    then
                        echo "Partition dir ${dir} does not exist. Creating it..."
                        mkdir -p ${dir}
                    fi

                    rm -f /scratch/aaji/temp/*
                    java ${jvm} -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.TestH1 $p ${osmmbb} ${dir}/osm ${temp} ${gridsize} ${alpha} ${ratio} ${sample}
                done    
            done < ${dir}/osm.${method}.txt

        fi
    done


