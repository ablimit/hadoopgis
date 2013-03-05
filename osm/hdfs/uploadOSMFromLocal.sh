#! /bin/bash

# planet as data source 1, europe as data source 2 
echo "uploading OSM planet .."
sudo -u hdfs hdfs dfs -copyFromLocal  /data2/ablimit/Data/spatialdata/osmout/planet.dat.1 /user/aaji/osm/planet.dat.1

echo "uploading OSM europe .."
sudo -u hdfs hdfs dfs -copyFromLocal  /data2/ablimit/Data/spatialdata/osmout/europe.dat.2 /user/aaji/osm/europe.dat.2

