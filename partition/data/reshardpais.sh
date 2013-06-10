#! /bin/bash

dataset=pais

# dir=/mnt/scratch1/aaji
dir=/scratch/aaji
opath=${dir}/partition
pardir=partres/${dataset}
parid=oc10000

if [ ! -e genpaismbb ] ;
then 
    echo "MBR generator is missing. "
    exit 1;
fi

for f in astroII.1 
# astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do
    echo "Generating MBRs for image ${f}"

    for seq in 1 2
    do
	echo "${seq}"
	# genpaismbb ${seq} < ${dir}/algo${seq}/${f}.markup.ablet.${seq} | python normalize.py pais > /dev/shm/mbb.${seq}.txt
    done

    # echo "Re-Partition the image .."
    for method in rtree 
# minskew rv rkHist sthist 
    do
	if [ ! -e ${opath}/${method}/${dataset}/${parid} ] ;
	then 
	    mkdir -p ${opath}/${method}/${dataset}/${parid}
	fi

	if [ -e ${pardir}/${parid}/${f}.${method}.txt ]
	then
	    echo "resharding image ${f} for ${method}"
	    cat /dev/shm/mbb.1.txt /dev/shm/mbb.2.txt | python genpid.py ${pardir}/${parid}/${f}.${method}.txt > a.txt 
	    cat a.txt | python reshardPAIS.py ${dir}/algo1/${f}.markup.ablet.1 ${dir}/algo2/${f}.markup.ablet.2 > ${opath}/${method}/${dataset}/${parid}/${f}.markup

	    # bzip2 > repart/pais/${f}.${method}.bz2
	fi
    done
done

