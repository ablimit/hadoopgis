#! /bin/bash


psql -d osm -c "BEGIN; ALTER TABLE planet_osm_polygon ADD COLUMN tilenameb varchar(32); UPDATE planet_osm_polygon SET tilenameb = tileb.tilename FROM tileb WHERE ST_Contains(tileb.mbb,planet_osm_polygon.way) = TRUE;  COMMIT; "

psql -d osmeu -c "BEGIN; ALTER TABLE planet_osm_polygon ADD COLUMN tilenameb varchar(32); UPDATE planet_osm_polygon SET tilenameb = tileb.tilename FROM tileb WHERE ST_Contains(tileb.mbb,planet_osm_polygon.way) = TRUE; COMMIT;"
