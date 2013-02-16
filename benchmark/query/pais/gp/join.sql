-- spatial join with predicate of polygon intersection


-- a) two way join on a single tile 

SELECT ST_Area(ST_Intersection(a.polygon, b.polygon)) / ST_Area(ST_Union( a.polygon, b.polygon))  AS a_ratio,
    ST_DISTANCE( ST_Centroid(b.polygon), ST_Centroid(a.polygon) ) AS centroid_distance
FROM pais.markup_polygon A, pais.markup_polygon B
WHERE  A.tilename ='gbm1.1-0000040960-0000040960' AND A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND 
    B.tilename ='gbm1.1-0000040960-0000040960' AND B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' AND
    ST_Intersects(A.polygon, B.polygon) = TRUE

-- b) two way join on a single image

SELECT ST_Area(ST_Intersection(a.polygon, b.polygon)) / ST_Area(ST_Union( a.polygon, b.polygon))  AS a_ratio,
    ST_DISTANCE( ST_Centroid(b.polygon), ST_Centroid(a.polygon) ) AS centroid_distance
FROM pais.markup_polygon A, pais.markup_polygon B
WHERE A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2'  AND 
    ST_Intersects(A.polygon, B.polygon) = TRUE

-- c) two way join on a collection

SELECT ST_Area(ST_Intersection(a.polygon, b.polygon)) / ST_Area(ST_Union( a.polygon, b.polygon))  AS a_ratio,
    ST_DISTANCE( ST_Centroid(b.polygon), ST_Centroid(a.polygon) ) AS centroid_distance
FROM pais.markup_polygon A, pais.markup_polygon B
WHERE A.tilename = B.tilename AND ST_Intersects(A.polygon, B.polygon) = TRUE ;


    
-- d) Find difference or missed nuclei between parameter 1 AND 2 of algorithm 'NS-MORPH' on a single tile 


SELECT 	count(*) 
FROM 		pais.markup_polygon
WHERE 	tilename ='gbm1.1-0000040960-0000040960' AND pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2'          
AND
markup_id NOT IN 
(
    SELECT 	B.markup_id
    FROM 		pais.markup_polygon A, pais.markup_polygon B
    WHERE  	A.tilename ='gbm1.1-0000040960-0000040960' AND A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND 
    B.tilename ='gbm1.1-0000040960-0000040960' AND B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' AND
    ST_intersects(A.polygon, B.polygon) = TRUE
);



-- e) Find how many markups in Set A with multiple intersects in Set B (Single Tile ).

SELECT  	A.pais_uid, A.tilename,  A.markup_id, count(*)
FROM 		pais.markup_polygon A, pais.markup_polygon B
WHERE  	A.tilename ='gbm1.1-0000040960-0000040960' AND A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND 
B.tilename ='gbm1.1-0000040960-0000040960' AND B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' AND
ST_intersects(A.polygon, B.polygon) = TRUE 
GROUP BY (A.pais_uid, A.tilename, A.markup_id) HAVING COUNT(*) > 1 ;

-- f) Find how many markups in Set A with multiple intersects in Set B (Single Tile ).

SELECT  	A.pais_uid, A.tilename,  A.markup_id, count(*)
FROM 		pais.markup_polygon A, pais.markup_polygon B
WHERE  	A.tilename ='gbm1.1-0000040960-0000040960' AND A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND 
B.tilename ='gbm1.1-0000040960-0000040960' AND B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' AND
ST_intersects(A.polygon, B.polygon) = TRUE 
GROUP BY (A.pais_uid, A.tilename, A.markup_id) HAVING COUNT(*) > 1 ;

