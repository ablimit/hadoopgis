#! /bin/bash

# jvm params 
jvm="-Xss4m -Xmx20G"

# file path
path=/data2/ablimit/Data/spatialdata
temp=/scratch/aaji/temp/
outpath=/scratch/aaji/osm
# partition parameters
gridsize=8
alpha=0.1
ratio=0.4
sample=0.5

if [ ! -e ${temp} ] ;
then
    echo "Temp ${temp} does not exist. Creating it..."
    mkdir -p ${temp}
fi


for size in 50000
do
    mark=${size%000}
    dir=data/partres/osm/oc${mark}k

    for method in rtree 
    do
	mbbdir=${outpath}/oc${mark}k/${method}

	if [ -e ${dir}/osm.${method}.txt ]; then

	    # mkdir -p ${mbbdir}
	    echo "MBR output dir: ${mbbdir}"

	    if [ ! -e ${outpath}/op/oc${mark}k ]; then
		mkdir -p ${outpath}/op/oc${mark}k 
	    fi

	    if [ ! -e ${outpath}/op/oc${mark}k/${method}.tsv.gz ] ; then 
		echo "oid -- pid matching ..."
		zcat ${outpath}/osm.mbb.filter.txt.gz | genpid ${dir}/osm.${method}.txt | gzip > ${outpath}/op/oc${mark}k/${method}.tsv.gz
	    fi
	    
	    if [ "$(ls -A $mbbdir)" ]; then
		echo "Directory ${mbbdir} is processed before."
	    else 
		echo "MBR hashing .."
		zcat ${outpath}/op/oc${mark}k/${method}.tsv.gz |  python subpart.py ${outpath}/osm.mbb.filter.txt.gz ${mbbdir}
	    fi
	    
	    for f in `ls -A ${mbbdir}`
	    do
		submbb=${mbbdir}/${f}
		echo "submbb file is: ${submbb}"
		lc=`wc -l ${submbb} | cut -d' ' -f1 `

		for subsize in  500 1000 2000 3000 4000 5000
		do
		    submark=${subsize%00}

		    subdir=data/partres/osm/oc${mark}k/${method}/oc${submark}c
		    if [ ! -e ${subdir} ] ;
		    then
			echo "Partition dir ${subdir} does not exist. Creating it..."
			mkdir -p ${subdir}
		    fi

		    # echo "calculating the partition layout.."
		    p=`expr $((lc/subsize))`

		    echo "partition params: |pid|=${f} |input|=${lc} |size|=${subsize} |p|=${p}"

		    if [ $p -ge 2 ]; then
			rm -f /scratch/aaji/temp/*
			java ${jvm} -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.TestH1 $p ${submbb} ${subdir}/osm.${f} ${temp} ${gridsize} ${alpha} ${ratio} ${sample}
		    fi
		done    
	    done
	fi
    done

done

