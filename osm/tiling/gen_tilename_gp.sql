-- 1) create colum tilename varchar 
-- 2) update 



BEGIN;
ALTER TABLE osm_polygon_planet ADD COLUMN "tilenameb" varchar(32);

--EXPLAIN UPDATE osm_polygon_planet 
--SET tilenameb = tileb.tilename
--FROM tileb
--WHERE ST_Contains(tileb.mbb,osm_polygon_planet.way) = TRUE;

UPDATE osm_polygon_planet 
SET tilenameb = tileb.tilename
FROM tileb
WHERE ST_Contains(tileb.mbb,osm_polygon_planet.way) = TRUE;
COMMIT;

BEGIN;
ALTER TABLE osm_polygon_europe ADD COLUMN "tilenameb" varchar(32);

UPDATE osm_polygon_europe
SET tilenameb = tileb.tilename
FROM tileb
WHERE ST_Contains(tileb.mbb,osm_polygon_europe.way) = TRUE;
COMMIT;

