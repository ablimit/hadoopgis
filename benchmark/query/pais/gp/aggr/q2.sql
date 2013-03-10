--c) spatial feature aggregation (single image)

SELECT 
    AVG(ST_Area(polygon)) AS AVG_AREA,
--    ST_Centroid(polygon) AS CENTROID,
--    ST_ConvexHull(polygon) AS CONVHULL,
    AVG(ST_Perimeter(polygon)) AS AVG_PERIMETER
FROM   	markup_polygon
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' ;
--GROUP BY pais_uid; 
