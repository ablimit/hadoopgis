
-- planet_tile4k4k data
BEGIN ;

DROP TABLE  osm_polygon_europe_tile4k4k; 

CREATE TABLE osm_polygon_planet_tile4k4k ( 
    id		integer,
    tilename	varchar(20),
    osm_id      bigint,
    z_order        integer) 
DISTRIBUTED BY (tilename);

SELECT AddGeometryColumn('public','osm_polygon_planet_tile4k4k', 'way', -1, 'POLYGON', 2);

COPY osm_polygon_planet_tile4k4k FROM '/data2/ablimit/Data/spatialdata/osmout/osm_polygon_planet.dat' WITH HEADER DELIMITER '|' ;

COMMIT ;


CREATE INDEX osm_polygon_planet_tile4k4k_sp_idx ON osm_polygon_planet_tile4k4k USING GIST (way);

CREATE INDEX osm_polygon_planet_tile4k4k_fidx ON osm_polygon_planet_tile4k4k (tilename);

VACUUM VERBOSE ANALYZE osm_polygon_planet_tile4k4k;


-- europe_tile4k4k data
-- tile level 
BEGIN ;

-- DROP TABLE  osm_polygon_europe_tile4k4k; 

CREATE TABLE osm_polygon_europe_tile4k4k ( 
    id		integer,
    tilename	varchar(20),
    osm_id      bigint,
    z_order        integer) 
DISTRIBUTED BY (tilename);

SELECT AddGeometryColumn('public','osm_polygon_europe_tile4k4k', 'way', -1, 'POLYGON', 2);

COPY osm_polygon_europe_tile4k4k FROM '/data2/ablimit/Data/spatialdata/osmout/osm_polygon_europe.dat' WITH HEADER DELIMITER '|' ;

-- COPY osm_polygon_europe_tile4k4k FROM '/tmp/temp.dat' WITH DELIMITER AS '|' CSV HEADER ;

COMMIT;


CREATE INDEX osm_polygon_europe_tile4k4k_sp_idx ON osm_polygon_europe_tile4k4k USING GIST (way);

CREATE INDEX osm_polygon_europe_tile4k4k_fidx ON osm_polygon_europe_tile4k4k (tilename);

VACUUM VERBOSE ANALYZE osm_polygon_europe_tile4k4k;


