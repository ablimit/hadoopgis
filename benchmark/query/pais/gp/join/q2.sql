-- b) two way join on a single image
\timing on 

SELECT ST_Area(ST_Intersection(A.polygon, B.polygon)) / ST_Area(ST_Union( A.polygon, B.polygon))  AS a_ratio,
    ST_DISTANCE( ST_Centroid(B.polygon), ST_Centroid(A.polygon) ) AS centroid_distance
FROM markup_polygon A, markup_polygon B
WHERE A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2'  AND 
      A.tilename = B.tilename AND ST_Intersects(A.polygon, B.polygon) = TRUE ;

