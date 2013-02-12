-- this script is used to generate a sequence id for planet_osm_polygon 

ALTER TABLE planet_osm_polygon ADD COLUMN "id" INTEGER;

CREATE SEQUENCE planet_osm_polygon_id_seq ;

UPDATE planet_osm_polygon SET id = nextval('planet_osm_polygon_id_seq');

ALTER TABLE planet_osm_polygon ALTER COLUMN "id" SET DEFAULT nextval('planet_osm_polygon_id_seq');

ALTER TABLE planet_osm_polygon ALTER COLUMN "id" SET NOT NULL;

ALTER TABLE planet_osm_polygon ADD UNIQUE ("id");

ALTER TABLE planet_osm_polygon DROP CONSTRAINT "planet_osm_polygon_id_key" RESTRICT;

ALTER TABLE planet_osm_polygon ADD PRIMARY KEY ("id");
