#!/bin/sh
#One-off task to create an osm database
DBNAME=osm
IMPORTOSM=/Users/ablimit/Documents/proj/Data/norway.osm.bz2

# if exists, remove the previous database
dropdb $DBNAME

# create routing database
createdb $DBNAME
# createlang plpgsql $DBNAME


# add PostGIS functions
echo "Adding PostGIS functions..."
psql -d $DBNAME -f /Library/PostgreSQL/9.2/share/postgresql/contrib/postgis/postgis.sql >/dev/null 2>&1 
psql -d $DBNAME -f /Library/PostgreSQL/9.2/share/postgresql/contrib/postgis/spatial_ref_sys.sql >/dev/null 2>&1 


echo "Importing osm file with osm2pgsql" >osm.log
echo "osm2pgsql -v --latlong --exclude-invalid-polygon --cache 50000 --style /home/aaji/softs/osm2pgsql/share/osm2pgsql/default.style --database $DBNAME --host node37 --port 5432 --number-processes 12 $IMPORTOSM "  >>macosm.log

date  >> macosm.log

osm2pgsql -v --latlong --exclude-invalid-polygon  --style /usr/local/share/osm2pgsql/default.style --database $DBNAME --port 5432 -U ablimit --number-processes 2 $IMPORTOSM #--number-processes 2 > /dev/null 2>&1 #--proj EPSG:4326

date >> macosm.log

