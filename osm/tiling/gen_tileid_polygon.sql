-- 1) create colum tilename varchar 
-- 2) update 

-- ALTER TABLE clean_osm ADD COLUMN "tilename" varchar;

EXPLAIN UPDATE clean_osm 
SET tilename = tile.tilename
FROM tile
WHERE ST_Contains(tile.mbb,clean_osm.way) = TRUE ;

UPDATE clean_osm 
SET tilename = tile.tilename
FROM tile
WHERE ST_Contains(tile.mbb,clean_osm.way) = TRUE ;
