#! /bin/bash


psql -d osm -c "ALTER TABLE planet_osm_polygon ADD COLUMN 'tilenameb' varchar(32);" 

psql -d osm -c " UPDATE planet_osm_polygon SET tilenameb = tileb.tilename FROM tileb WHERE ST_Contains(tileb.mbb,osm_polygon_planet.way) = TRUE;"

psql -d osmeu -c "ALTER TABLE planet_osm_polygon ADD COLUMN 'tilenameb' varchar(32);" 

psql -d osmeu -c " UPDATE planet_osm_polygon SET tilenameb = tileb.tilename FROM tileb WHERE ST_Contains(tileb.mbb,osm_polygon_planet.way) = TRUE;"
