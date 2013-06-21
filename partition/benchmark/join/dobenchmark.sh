#! /bin/bash


for job in pais osm
do

    echo "Job: ${job} "
    cd ${job}
    sh runMapReduce.sh
    cd -
done

