-- feature aggreagation

-- For each human marked region (polygon), count the number of nuclei in that region and the average features of them.

-- a) flat feature aggregation  with join
/* 
SELECT 	count(m.markup_id) NUM_NUCLEI, 
    avg(f.AREA) AVG_AREA,
    avg(f.eccentricity) AVG_ECC,
    avg(f.PERIMETER) AVG_PERIMETER
FROM   	PAIS.MARKUP_POLYGON m, 
        PAIS.MARKUP_POLYGON_HUMAN r,
        PAIS.CALCULATION_FLAT f
WHERE  	m.pais_uid ='gbm1.1_40x_20x_NS-MORPH_1' AND 
        f.pais_uid = m.pais_uid AND 
        r.pais_uid = 'gbm1.1_40x_40x_RG-HUMAN_1' AND 
        f.pais_uid = r.pais_uid AND 
        f.markup_id = m.markup_id AND 
        ST_Contains(r.polygon, m.polygon) = TRUE;
*/

-- b) spatial feature aggregation (tile)
SELECT 
    ST_Area(polygon) AS AREA,
    ST_Centroid(polygon) AS CENTROID,
    ST_ConvexHull(polygon) AS CONVHULL,
    ST_Perimeter(polygon) AS PERIMETER
FROM   	pais.markup_polygon
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND tilename ='gbm1.1-0000040960-0000040960' ;
-- ST_Contains( ST_PolygonFromText('POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))', 100), polygon ) = TRUE ;


--c) spatial feature aggregation (image)
SELECT 
    ST_Area(polygon) AS AREA,
    ST_Centroid(polygon) AS CENTROID,
    ST_ConvexHull(polygon) AS CONVHULL,
    ST_Perimeter(polygon) AS PERIMETER
FROM   	pais.markup_polygon
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'; 
-- ST_Contains( ST_PolygonFromText('POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))', 100), polygon ) = TRUE ;

--d) spatial feature aggregation (collection)
SELECT 
    ST_Area(polygon) AS AREA,
    ST_Centroid(polygon) AS CENTROID,
    ST_ConvexHull(polygon) AS CONVHULL,
    ST_Perimeter(polygon) AS PERIMETER
FROM   	pais.markup_polygon ;
-- ST_Contains( ST_PolygonFromText('POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))', 100), polygon ) = TRUE ;
