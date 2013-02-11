
BEGIN ;
CREATE TABLE osm_polygons (
    osm_id  bigint, 
    z_order real,
    area  real);

SELECT AddGeometryColumn('osm_polygons','way', 4326, 'POLYGON', 2 ) ;

COPY (SELECT osm_id, z_order,area, ST_AsText(way) AS way FROM planet_osm_polygon LIMIT 1000) TO '/dev/shm/dump.dat' WITH DELIMITER AS '|' CSV HEADER ;

COPY osm_polygons( osm_id,z_order,area, way) FROM '/dev/shm/dump.dat' WITH DELIMITER AS '|' CSV HEADER ;

-- SELECT osm_id,z_order,area,way INTO osm_polygons FROM planet_osm_polygon ; 

COMMIT ;
