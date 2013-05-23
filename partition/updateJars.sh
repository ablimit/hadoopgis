#! /bin/bash


loc=/home/aaji/proj/hadoopgis/xxl/xxlcore/target

for file in xxlcore-2.1-SNAPSHOT.jar xxlcore-2.1-SNAPSHOT-tests.jar
do
    if [ -f "${loc}/${file}" ]
    then
        echo "$file found and copied."
        cp ${loc}/${file} /home/aaji/proj/hadoopgis/partition/
    else
        echo "$file not found."
    fi
done


