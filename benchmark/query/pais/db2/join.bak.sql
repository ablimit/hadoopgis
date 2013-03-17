-- spatial join with predicate of polygon intersection

-- a) two way join
SELECT count(*) from (

    SELECT A.pais_uid, A.tilename,  
    CAST(  db2gse.ST_Area(db2gse.ST_Intersection(a.polygon, b.polygon) ) / db2gse.ST_Area(db2gse.ST_UNION( a.polygon, b.polygon))  AS DECIMAL(4,2) )AS area_ratio,  
    CAST( DB2GSE.ST_DISTANCE( db2gse.ST_Centroid(b.polygon), db2gse.ST_Centroid(a.polygon) )  AS DECIMAL(5,2) ) AS centroid_distance
    FROM pais.markup_polygon A, pais.markup_polygon B
    WHERE  A.tilename ='gbm1.1-0000040960-0000040960' and A.pais_uid =       		'gbm1.1_40x_20x_NS-MORPH_1' and 
    B.tilename ='gbm1.1-0000040960-0000040960' and B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' and
    DB2GSE.ST_intersects(A.polygon, B.polygon) = 1 
    selectivity 0.0001 WITH UR

); 


--b) Find difference or missed nuclei between parameter 1 and 2 of algorithm 'NS-MORPH'.


SELECT 	count(*) 
FROM 		pais.markup_polygon
WHERE 	tilename ='gbm1.1-0000040960-0000040960' and pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2'          
AND
markup_id NOT IN 
(
    SELECT 	B.markup_id
    FROM 		pais.markup_polygon A, pais.markup_polygon B
    WHERE  	A.tilename ='gbm1.1-0000040960-0000040960' and A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' and 
    B.tilename ='gbm1.1-0000040960-0000040960' and B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' and
    DB2GSE.ST_intersects(A.polygon, B.polygon) = 1 
    selectivity 0.0001 WITH UR
);



-- c) Find how many markups in Set A with multiple intersects in Set B.

SELECT  	A.pais_uid, A.tilename,  A.markup_id, count(*)
FROM 		pais.markup_polygon A, pais.markup_polygon B
WHERE  	A.tilename ='gbm1.1-0000040960-0000040960' and A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' and 
B.tilename ='gbm1.1-0000040960-0000040960' and B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' and
DB2GSE.ST_intersects(A.polygon, B.polygon) = 1 selectivity 0.0001 
GROUP BY (A.pais_uid, A.tilename, A.markup_id) HAVING COUNT(*) > 1 WITH UR;

SELECT current timestamp FROM sysibm.sysdummy1;
SELECT COUNT(*) FROM (SELECT 
A.pais_uid
	,A.tilename
--,DECIMAL(db2gse.ST_Area(a.polygon) / db2gse.ST_Area(b.polygon), 5, 2)
	,
--DECIMAL(SQRT((a.centroid_x - b.centroid_x) * (a.centroid_x - b.centroid_x) + (a.centroid_y - b.centroid_y) * (a.centroid_y - b.centroid_y)), 5, 2) AS xy_distance
	CAST(  db2gse.ST_Area(db2gse.ST_Intersection(a.polygon, b.polygon) )/db2gse.ST_Area(db2gse.ST_UNION( a.polygon, b.polygon))  AS DECIMAL(4,2) ) AS area_ratio 
	FROM pais.markup_polygon A, pais.markup_polygon B
	WHERE  A.tilename =B.tilename  
	and      A.pais_uid = 'oligoIII.2_40x_20x_NS-MORPH_1' 
	and      B.pais_uid = 'oligoIII.2_40x_20x_NS-MORPH_2' 
	and      DB2GSE.ST_intersects(A.polygon, B.polygon) = 1 
	selectivity 0.000001
	-- FETCH FIRST 100000 ROWS ONLY
	WITH UR )
	;

SELECT current timestamp FROM sysibm.sysdummy1;
