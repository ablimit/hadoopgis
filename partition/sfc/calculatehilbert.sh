#! /bin/bash

# jvm params 
jvm="-Xss4m -Xmx5G"

# file path
path=/data2/ablimit/Data/spatialdata/bakup/data/osm.mbb.norm.filter.dat
outpath=/data2/ablimit/Data/spatialdata/bakup/data/partition/osm/hilbert/osm.mbb.hilbert.dat
prec=20

cp /home/aaji/proj/hadoopgis/xxl/xxlcore/target/*.jar ./

# head -n 10000 ${path} > temp.dat 

java ${jvm} -cp xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. xxl.core.spatial.Hilbert ${path} ${prec} > ${outpath} 

