#! /bin/bash

opath=/mnt/scratch1/aaji/fixedgrid/pais/wbo
ipath=/mnt/scratch1/aaji/algo

for size in 128 256 512 768 1024 2048 4096 8192 16384
do

    if [ ! -e ${opath}/grid${size} ] ;
    then 
	mkdir -p ${opath}/grid${size}

	xsplit=`expr $((110592/size))`
	ysplit=`expr $((57344/size))`

	echo "Grid size: ${xsplit}x${ysplit}"

	for f in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
	do
	    echo "FixedGrid for $f: "

	    for algo in 1 2
	    do
		./mapper  -p $f -w 0 -s 0 -n 57344 -e 110592 -x ${xsplit} -y ${ysplit} -d pais < ${ipath}${algo}/${f}.markup.ablet.${algo} > ${opath}/grid${size}/${f}.markup.${algo}
	    done
	done
    fi
done

