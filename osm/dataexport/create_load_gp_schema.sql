
-- tile level 
BEGIN ;

-- DROP TABLE  osm_polygon_europe; 

CREATE TABLE osm_polygon_europe ( 
    id		integer,
    tilename	varchar(20),
    osm_id      bigint,
    z_order        integer) 
DISTRIBUTED BY (tilename);

SELECT AddGeometryColumn('public','osm_polygon_europe', 'way', -1, 'POLYGON', 2);

COPY osm_polygon_europe FROM '/data2/ablimit/Data/spatialdata/osmout/europe_osm_polygon.dat' WITH HEADER DELIMITER '|' ;

-- COPY osm_polygon_europe FROM '/tmp/temp.dat' WITH DELIMITER AS '|' CSV HEADER ;

COMMIT ;


CREATE INDEX osm_polygon_europe_sp_idx ON osm_polygon_europe USING GIST (way);

CREATE INDEX osm_polygon_europe_fidx ON osm_polygon_europe (tilename);

VACUUM VERBOSE ANALYZE osm_polygon_europe;

