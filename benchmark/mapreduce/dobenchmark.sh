#! /bin/bash


for job in pais osm 
do

    echo "Job: ${job} "
    cd ${job}
    sh dobenchmark.sh
    cd -
done

