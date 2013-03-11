--d) spatial feature aggregation (collection)

SELECT 
    AVG(ST_Area(polygon)) AS AVG_AREA,
--    ST_Centroid(polygon) AS CENTROID,
--    ST_ConvexHull(polygon) AS CONVHULL,
    AVG(ST_Perimeter(polygon)) AS AVG_PERIMETER
FROM   	markup_polygon;

