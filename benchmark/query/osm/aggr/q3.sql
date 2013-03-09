-- feature aggreagation
\timing on

--e) spatial feature aggregation with spatial predicate (collection)
SELECT 
    ST_Area(way) AS AREA,
    ST_Centroid(way) AS CENTROID,
    ST_ConvexHull(way) AS CONVHULL,
    ST_Perimeter(way) AS PERIMETER
FROM osm_polygon_planet
WHERE ST_Area(way) > 0.5 GROUP BY tilename;

