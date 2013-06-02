#! /bin/bash

for seq in 1 2
do
    for f in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
    do
        echo "Generating MBRs for image ${f}"
        genmbb < algo${seq}/${f}.markup.ablet.${seq} | python normolize.py pais > /dev/shm/mbb.txt

        for method in rtree minskew rv rkHist sthist 
        do
            if [ -e partres/${f}.${method}.txt ]
            then
                echo "resharding image ${f} for ${method}"
                python genpid.py partres/${f}.${method}.txt < /dev/shm/mbb.txt  | python reshardPAIS.py algo${seq}/${f}.markup.ablet.${seq} | bzip2 > repart/algo${seq}/${f}.${method}.${seq}.bz2
            fi
        done
    done
done

