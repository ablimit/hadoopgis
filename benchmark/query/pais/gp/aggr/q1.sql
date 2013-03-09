\timing on 
-- b) spatial feature aggregation (single tile)

SELECT 
    ST_Area(polygon) AS AREA,
    ST_Centroid(polygon) AS CENTROID,
    ST_ConvexHull(polygon) AS CONVHULL,
    ST_Perimeter(polygon) AS PERIMETER
FROM   	markup_polygon
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND tilename ='gbm1.1-0000040960-0000040960' ;

