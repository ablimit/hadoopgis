-- feature aggreagation

-- For each human marked region (polygon), count the number of nuclei in that region and the average features of them.

-- a) flat feature aggregation  with join
 
-- SELECT 	count(m.markup_id) NUM_NUCLEI, 
--    avg(f.AREA) AVG_AREA,
--    avg(f.eccentricity) AVG_ECC,
--    avg(f.PERIMETER) AVG_PERIMETER
-- FROM   	PAIS.MARKUP_POLYGON m, 
--        PAIS.MARKUP_POLYGON_HUMAN r,
--        PAIS.CALCULATION_FLAT f
-- WHERE  	m.pais_uid ='gbm1.1_40x_20x_NS-MORPH_1' AND 
--	  f.pais_uid = m.pais_uid AND 
--        r.pais_uid = 'gbm1.1_40x_40x_RG-HUMAN_1' AND 
--        f.pais_uid = r.pais_uid AND 
--        f.markup_id = m.markup_id AND 
--        ST_Contains(r.polygon, m.polygon) = TRUE;

\timing on 
-- b) spatial feature aggregation (single tile)

SELECT 
    ST_Area(polygon) AS AREA,
    ST_Centroid(polygon) AS CENTROID,
    ST_ConvexHull(polygon) AS CONVHULL,
    ST_Perimeter(polygon) AS PERIMETER
FROM   	markup_polygon
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND tilename ='gbm1.1-0000040960-0000040960' ;


--c) spatial feature aggregation (single image)
SELECT 
    ST_Area(polygon) AS AREA,
    ST_Centroid(polygon) AS CENTROID,
    ST_ConvexHull(polygon) AS CONVHULL,
    ST_Perimeter(polygon) AS PERIMETER
FROM   	markup_polygon
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'; 

--d) spatial feature aggregation (collection)
SELECT 
    ST_Area(polygon) AS AREA,
    ST_Centroid(polygon) AS CENTROID,
    ST_ConvexHull(polygon) AS CONVHULL,
    ST_Perimeter(polygon) AS PERIMETER
FROM   	markup_polygon ;


--e) spatial feature aggregation with spatial predicate (collection)
SELECT 
    ST_Area(polygon) AS AREA,
    ST_Centroid(polygon) AS CENTROID,
    ST_ConvexHull(polygon) AS CONVHULL,
    ST_Perimeter(polygon) AS PERIMETER
FROM	markup_polygon 
WHERE ST_Area(polygon) > 125.0;

\timing off

