
CREATE TABLE clean_polygon (osm_id  bigint, z_order real, area  real, );

SELECT AddGeometryColumn('clean_polygon','way', 4326, 'POLYGON', 2 ) ;

SELECT osm_id,z_order,area,way INTO clean_osm FROM planet_osm_polygon ; 


