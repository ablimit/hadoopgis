-- spatial join with predicate of polygon intersection

-- a) two way join
SELECT count(*) from (

    SELECT A.id, A.tilename, ST_Area(ST_Intersection(A.way, B.way)) / ST_Area(ST_Union( A.way, B.way))  AS a_ratio,
    ST_DISTANCE(ST_Centroid(A.way), ST_Centroid(B.way)) AS centroid_distance
    FROM osm_polygon_planet_fourxfour A, osm_polygon_europe_fourxfour B
    WHERE  A.tilename = B.tilename AND length(A.tilename) > 2 AND 
	ST_Equals(A.way, B.way) = FALSE
) AS temp ; 


