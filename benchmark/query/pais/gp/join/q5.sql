-- e) Find how many markups in Set A with multiple intersects in Set B (Single Tile ).
\timing on

SELECT  	A.pais_uid, A.tilename,  A.markup_id, count(*)
FROM 		markup_polygon A, markup_polygon B
WHERE  	A.tilename ='gbm1.1-0000040960-0000040960' AND A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND 
B.tilename ='gbm1.1-0000040960-0000040960' AND B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' AND
ST_intersects(A.polygon, B.polygon) = TRUE 
GROUP BY (A.pais_uid, A.tilename, A.markup_id) HAVING COUNT(*) > 1 ;
