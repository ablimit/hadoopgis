#!/bin/sh
#One-off task to create an osm database
DBNAME=osmeu
IMPORTOSM=/data2/ablimit/Data/spatialdata/osm/europe2011/europe.osm.bz2
# IMPORTOSM=/data2/ablimit/Data/spatialdata/osm/planet-latest.osm.bz2
#IMPORTOSM=/data2/ablimit/Data/spatialdata/osm/europe.osm.bz2

# if exists, remove the previous database
echo "start parsing europe data from 2011...." >> osm.log
dropdb $DBNAME

createdb $DBNAME


# add PostGIS functions
echo "Adding PostGIS functions..."
psql -d $DBNAME -f /home/aaji/softs/share/postgresql/contrib/postgis-1.5/postgis.sql >/dev/null 2>&1 
psql -d $DBNAME -f /home/aaji/softs/share/postgresql/contrib/postgis-1.5/spatial_ref_sys.sql >/dev/null 2>&1 


echo "Importing osm file with osm2pgsql" >>osm.log
echo "osm2pgsql -v --latlong --exclude-invalid-polygon --cache 50000 --style /home/aaji/softs/osm2pgsql/share/osm2pgsql/default.style --database $DBNAME --host node37 --port 5432 --number-processes 12 $IMPORTOSM "  >>osm.log

date  >> osm.log

osm2pgsql -v --latlong --exclude-invalid-polygon --cache 50000 --style /home/aaji/softs/osm2pgsql/share/osm2pgsql/default.style --database $DBNAME --host node37 --port 5432 --number-processes 12 $IMPORTOSM #--number-processes 2 > /dev/null 2>&1 #--proj EPSG:4326


date >> osm.log

