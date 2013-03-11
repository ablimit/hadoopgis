--d) spatial feature aggregation with spatial predicate (collection)

SELECT 
    AVG(ST_Area(polygon)) AS AVG_AREA,
--    ST_Centroid(polygon) AS CENTROID,
--    ST_ConvexHull(polygon) AS CONVHULL,
    AVG(ST_Perimeter(polygon)) AS AVG_PERIMETER
FROM   	markup_polygon 
WHERE ST_Area(polygon) > 125.0
GROUP BY pais_uid;

