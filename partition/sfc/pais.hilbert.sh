#! /bin/bash

# jvm params 
jvm="-Xss4m -Xmx1024m"
jlib=xxlcore-2.1-SNAPSHOT.jar:xxlcore-2.1-SNAPSHOT-tests.jar:. 

# file path
ipath=/home/aaji/temp
prec=20

# cp /home/aaji/proj/hadoopgis/xxl/xxlcore/target/*.jar ./
# head -n 10000 ${path} > temp.dat 

# for image in astroII.1 astroII.2 gbm0.1 gbm0.2 gbm1.1 gbm1.2 gbm2.1 gbm2.2 normal.2 normal.3 oligoastroII.1 oligoastroII.2 oligoastroIII.1 oligoastroIII.2 oligoII.1 oligoII.2 oligoIII.1 oligoIII.2
for image in oligoIII.2
do
  echo "processing image: ${image}"
  java ${jvm} -cp ${jlib} xxl.core.spatial.Hilbert ${ipath}/mbb/${image}.norm.1.dat ${prec} > ${ipath}/hc/${image}.1.dat
done

