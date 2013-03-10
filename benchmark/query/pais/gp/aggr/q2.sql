--c) spatial feature aggregation (single image)
\timing on 
SELECT 
    ST_Area(polygon) AS AREA,
    ST_Centroid(polygon) AS CENTROID,
    ST_ConvexHull(polygon) AS CONVHULL,
    ST_Perimeter(polygon) AS PERIMETER
FROM   	markup_polygon
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'; 
