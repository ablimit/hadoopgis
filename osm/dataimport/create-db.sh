# run this as postgres user, eg:
# imposm-psqldb > create_db.sh; sudo su postgres; sh ./create_db.sh

postgisloc=/home/aaji/softs/share/postgresql/contrib/postgis-1.5

set -xe
# createuser --no-superuser --no-createrole --createdb aaji
createdb -E UTF8 -O aaji osmplanet
# createlang plpgsql osmplanet
psql -d osmplanet -f $postgisloc/postgis.sql 				# <- CHANGE THIS PATH
psql -d osmplanet -f $postgisloc/spatial_ref_sys.sql 			# <- CHANGE THIS PATH
psql -d osmplanet -f /home/aaji/softs/python/lib/python2.7/site-packages/imposm-2.5.0-py2.7-linux-x86_64.egg/imposm/900913.sql

echo "ALTER TABLE geometry_columns OWNER TO aaji;"
echo "ALTER TABLE geometry_columns OWNER TO aaji;" | psql -d osmplanet

echo "ALTER TABLE spatial_ref_sys OWNER TO aaji;"
echo "ALTER TABLE spatial_ref_sys OWNER TO aaji;" | psql -d osmplanet

echo "ALTER USER aaji WITH PASSWORD 'osm';"
echo "ALTER USER aaji WITH PASSWORD 'osm';" |psql -d osmplanet
# echo "host	osmplanet	aaji	127.0.0.1/32	md5" >> /data2/ablimit/Data/spatialdata/data/pg_hba.conf

set +x
# echo "Done. Don't forget to restart postgresql!"
