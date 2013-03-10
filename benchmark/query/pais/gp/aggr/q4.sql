--e) spatial feature aggregation with spatial predicate (collection)
\timing on
SELECT 
    ST_Area(polygon) AS AREA,
    ST_Centroid(polygon) AS CENTROID,
    ST_ConvexHull(polygon) AS CONVHULL,
    ST_Perimeter(polygon) AS PERIMETER
FROM	markup_polygon 
WHERE ST_Area(polygon) > 125.0;

\timing off

