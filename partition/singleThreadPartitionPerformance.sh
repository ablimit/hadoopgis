#! /bin/bash


for script in pais osm
do
  for loc in fg sfc strip rplus rtree
  do
    pushd ${loc}/serial ;
    echo "${script}----------------------------"
    echo "${loc}-------------------------------"
    echo "-------------------------------------"
    sh ${script}.pipeline.sh > ${script}.log
    popd
  done
done
