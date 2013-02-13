-- spatial join with predicate of polygon intersection

-- a) two way join
SELECT count(*) from (

    SELECT A.id, A.tilename, ST_Area(ST_Intersection(A.way, B.way)) / ST_Area(ST_Union( A.way, B.way))  AS a_ratio,
    ST_DISTANCE(ST_Centroid(A.way), ST_Centroid(B.way)) AS centroid_distance
    FROM osm.osm_polygon_planet A, osm.osm_polygon_europe B
    WHERE  A.tilename = B.tilename AND ST_Equals(A.polygon, B.polygon) = FALSE;
) AS temp ; 

