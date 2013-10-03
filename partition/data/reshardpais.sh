#! /bin/bash

dataset=pais

dir=/mnt/scratch1/aaji # /mnt/scratch1/aaji/partition/rtree/pais/oc10000/
# dir=/scratch/aaji
opath=${dir}/partition
pardir=partres/${dataset}
parid=oc10000

if [ ! -e genpaismbb ] ;
then 
    echo "MBR generator is missing. "
    exit 1;
fi

for f in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
do

    echo "processing image ${f}"

    for size in 50 100 150 250 500 750 1000 1500 2000 #2500 5000 10000 15000 20000 25000 30000 50000

    do
	parid="oc${size}"
	echo "shardset: ${parid}"
	echo -e "\tstep 1: MBR"
	for seq in 1 2
	do
	    # echo "${seq}"
	    genpaismbb ${seq} < ${dir}/algo${seq}/${f}.markup.ablet.${seq} | python normalize.py pais > /dev/shm/mbb.${seq}.txt
	done


	echo -e "\tstep 2: repartition"
	for method in rtree minskew rv rkHist sthist 
	do
	    if [ ! -e ${opath}/${dataset}/${method}/${parid} ] ;
	    then 
		mkdir -p ${opath}/${dataset}/${method}/${parid}
	    fi

	    if [ -e ${pardir}/${parid}/${f}.${method}.txt ]
	    then
		# echo "resharding image ${f} for ${method}"
		echo -e "\t\tmethod: ${method}"
		cat /dev/shm/mbb.1.txt /dev/shm/mbb.2.txt | python genpid.py ${pardir}/${parid}/${f}.${method}.txt | python reshardPAIS.py ${dir}/algo1/${f}.markup.ablet.1 ${dir}/algo2/${f}.markup.ablet.2 > ${opath}/${dataset}/${method}/${parid}/${f}.markup

		# bzip2 > repart/pais/${f}.${method}.bz2
	    else 
		for seq in 1 2
		do
		    python addsidPAIS.py ${seq} < ${dir}/algo${seq}/${f}.markup.ablet.${seq} >> ${opath}/${dataset}/${method}/${parid}/${f}.markup

		done
	    fi
	done

    done
done

