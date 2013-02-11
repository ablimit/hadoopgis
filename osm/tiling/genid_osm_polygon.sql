-- this script is used to generate a sequence id for clean_osm 

ALTER TABLE clean_osm ADD COLUMN "id" INTEGER;

CREATE SEQUENCE clean_osm_id_seq ;

UPDATE clean_osm SET id = nextval('clean_osm_id_seq');

ALTER TABLE clean_osm ALTER COLUMN "id" SET DEFAULT nextval('clean_osm_id_seq');

ALTER TABLE clean_osm ALTER COLUMN "id" SET NOT NULL;

ALTER TABLE clean_osm ADD UNIQUE ("id");

ALTER TABLE clean_osm DROP CONSTRAINT "clean_osm_id_key" RESTRICT;

ALTER TABLE clean_osm ADD PRIMARY KEY ("id");
