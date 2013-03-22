#! /bin/bash

for task in containment aggregation #join
do

    echo "Task: ${task} "
    cd ${task}
    sh runMapReduce.sh 6
    cd -
done

