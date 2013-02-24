
-- planet_fourxfour data
BEGIN ;

--DROP TABLE  osm_polygon_europe_fourxfour; 

CREATE TABLE osm_polygon_planet_fourxfour ( 
    id		integer,
    tilename	varchar(32),
    osm_id      bigint,
    z_order        integer) 
DISTRIBUTED BY (tilename);

SELECT AddGeometryColumn('public','osm_polygon_planet_fourxfour', 'way', -1, 'POLYGON', 2);

COPY osm_polygon_planet_fourxfour FROM '/data2/ablimit/Data/spatialdata/osmout/osm_polygon_planet.dat' WITH HEADER DELIMITER '|' ;

CREATE INDEX osm_polygon_planet_fourxfour_sp_idx ON osm_polygon_planet_fourxfour USING GIST (way);

CREATE INDEX osm_polygon_planet_fourxfour_fidx ON osm_polygon_planet_fourxfour (tilename);

COMMIT ;

VACUUM VERBOSE ANALYZE osm_polygon_planet_fourxfour;


-- europe_fourxfour data
-- tile level 
BEGIN ;

-- DROP TABLE  osm_polygon_europe_fourxfour; 

CREATE TABLE osm_polygon_europe_fourxfour ( 
    id		integer,
    tilename	varchar(32),
    osm_id      bigint,
    z_order        integer) 
DISTRIBUTED BY (tilename);

SELECT AddGeometryColumn('public','osm_polygon_europe_fourxfour', 'way', -1, 'POLYGON', 2);

COPY osm_polygon_europe_fourxfour FROM '/data2/ablimit/Data/spatialdata/osmout/osm_polygon_europe.dat' WITH HEADER DELIMITER '|' ;

-- COPY osm_polygon_europe_fourxfour FROM '/tmp/temp.dat' WITH DELIMITER AS '|' CSV HEADER ;

CREATE INDEX osm_polygon_europe_fourxfour_sp_idx ON osm_polygon_europe_fourxfour USING GIST (way);

CREATE INDEX osm_polygon_europe_fourxfour_fidx ON osm_polygon_europe_fourxfour (tilename);

COMMIT;

VACUUM VERBOSE ANALYZE osm_polygon_europe_fourxfour;

