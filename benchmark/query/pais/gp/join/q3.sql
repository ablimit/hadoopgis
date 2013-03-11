-- c) two way join on a collection
\timing on 

SELECT ST_Area(ST_Intersection(A.polygon, B.polygon)) / ST_Area(ST_Union( A.polygon, B.polygon))  AS a_ratio,
    ST_DISTANCE( ST_Centroid(A.polygon), ST_Centroid(B.polygon) ) AS centroid_distance
FROM markup_polygon A, markup_polygon B
WHERE A.tilename = B.tilename AND A.paisST_Intersects(A.polygon, B.polygon) = TRUE ;

