
-- tile level 
BEGIN ;

DROP TABLE  osm_polygon; 

CREATE TABLE osm_polygon ( 
    id		integer,
    tilename	varchar(20),
    osm_id      bigint,
    z_order        integer) 
DISTRIBUTED BY (tilename);

SELECT AddGeometryColumn('public','osm_polygon', 'way', -1, 'POLYGON', 2);

COPY osm_polygon FROM '/data2/ablimit/Data/spatialdata/osmout/clean_osm.dat' WITH HEADER DELIMITER '|' ;
-- COPY osm_polygon FROM '/tmp/temp.dat' WITH DELIMITER AS '|' CSV HEADER ;

COMMIT ;

CREATE INDEX osm_polygon_sp_idx ON osm_polygon USING GIST (way);

CREATE INDEX osm_polygon_fidx ON osm_polygon (tilename);

VACUUM VERBOSE ANALYZE osm_polygon;

