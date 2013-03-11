-- d) Find difference or missed nuclei between parameter 1 AND 2 of algorithm 'NS-MORPH' on a single tile 

\timing on 

SELECT 	count(*) 
FROM 		markup_polygon
WHERE 	tilename ='gbm1.1-0000040960-0000040960' AND pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2'          
AND
markup_id NOT IN 
(
    SELECT 	B.markup_id
    FROM 		markup_polygon A, markup_polygon B
    WHERE  	A.tilename ='gbm1.1-0000040960-0000040960' AND A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND 
    B.tilename ='gbm1.1-0000040960-0000040960' AND B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' AND
    ST_intersects(A.polygon, B.polygon) = TRUE
);

