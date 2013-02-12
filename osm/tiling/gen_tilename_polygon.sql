-- 1) create colum tilename varchar 
-- 2) update 

ALTER TABLE planet_osm_polygon ADD COLUMN "tilename" varchar(32);

EXPLAIN UPDATE planet_osm_polygon
SET tilename = tile.tilename
FROM tile
WHERE ST_Contains(tile.mbb,planet_osm_polygon.way) = TRUE ;


UPDATE planet_osm_polygon 
SET tilename = tile.tilename
FROM tile
WHERE ST_Contains(tile.mbb,planet_osm_polygon.way) = TRUE ;
