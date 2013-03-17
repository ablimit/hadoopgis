#! /bin/bash

inloc=/data2/ablimit
outloc=/data2/ablimit/Data/spatialdata/pais/boundarytest

make -f Makefile regen

myhost='node37.clus.cci.emory.edu'

if [ "$HOSTNAME" = ${myhost} ] ; then

    for fac in 2 3 4 5 6 7 8 9
    do
        echo "Duplication Factor = ${fac}"
        for i in 1 2
        do
            echo "ALGO ID = ${i}"
            for image in `cat imagelist`
            do
                echo "Duplicating Image -- ${image}"
                echo ""
                mkdir -p ${outloc}/rep${fac}/algo${i}
                ./regen 2048 ${fac}  /tmp/abc_${i}_${j}.txt < ${inloc}/algo${i}/${image}.markup.ablet.${i} > ${outloc}/rep${fac}/algo${i}/${image}.markup.ablet.${i}
            done
        done

    done


else
    echo "You can not execute deduplication on this host"
    exit
fi

