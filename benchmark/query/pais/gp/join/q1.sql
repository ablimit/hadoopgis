-- spatial join with predicate of polygon intersection

-- a) two way join on a single tile 
\timing on 

SELECT ST_Area(ST_Intersection(A.polygon, B.polygon)) / ST_Area(ST_Union( A.polygon, B.polygon))  AS a_ratio,
    ST_DISTANCE( ST_Centroid(A.polygon), ST_Centroid(B.polygon) ) AS centroid_distance
FROM markup_polygon A, markup_polygon B
WHERE  A.tilename ='gbm1.1-0000040960-0000040960' AND A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND 
    B.tilename ='gbm1.1-0000040960-0000040960' AND B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' AND
    ST_Intersects(A.polygon, B.polygon) = TRUE ;

