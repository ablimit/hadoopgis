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
sample=0.4

if [ ! -e ${temp} ] ;
then
    echo "Temp ${temp} does not exist. Creating it..."
    mkdir -p ${temp}
fi


for size in 50000 #100000 200000 300000 400000 500000 
do
    mark=${size%000}
    dir=data/partres/osm/oc${mark}k

    for method in rtree #minskew minskewrefine rv rkHist
    do
	mbbdir=${outpath}/oc${mark}k/${method}

	if [ -e ${dir}/osm.${method}.txt ] ; then
	    # && [ ! -e ${mbbdir} ] ; then

	    # mkdir -p ${mbbdir}
	    echo "MBR output dir: ${mbbdir}"

	    if [ ! -e ${outpath}/op/oc${mark}k ]; then
		mkdir -p ${outpath}/op/oc${mark}k 
	    fi

	    echo "oid -- pid matching ..."
	    zcat ${outpath}/osm.mbb.filter.txt.gz | genpid ${dir}/osm.${method}.txt | gzip > ${outpath}/op/oc${mark}k/${method}.tsv.gz
	    echo "MBR hashing .."
	    zcat ${outpath}/op/oc${mark}k/${method}.tsv.gz |  python subpart.py ${outpath}/osm.mbb.filter.txt.gz ${mbbdir}

	    for f in `ls ${mbbdir}`
	    do
		submbb=${mbbdir}/${f}
		echo "submbb file is: ${submbb}"

		for subsize in  500 #1000 2000 3000 4000 5000
		do
		    submark=${subsize%00}
		    echo "calculating the partition layout.."
		    lc=`wc -l ${submbb} | cut -d' ' -f1 `
		    p=`expr $((lc/subsize))`

		    echo "partition params: |pid|=${f} |input|=${lc} |size|=${subsize} |p|=${p}"

		    if [ $p -ge 2 ]; then

			#  c for centum
			subdir=data/partres/osm/oc${mark}k/${method}/oc${submark}c
			if [ ! -e ${subdir} ] ;
			then
			    echo "Partition dir ${subdir} does not exist. Creating it..."
			    mkdir -p ${subdir}
			fi

			rm -f /scratch/aaji/temp/*
			java ${jvm} -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.TestH1 $p ${submbb} ${subdir}/osm ${temp} ${gridsize} ${alpha} ${ratio} ${sample}
		    fi
		done    
	    done
	fi
    done

done

