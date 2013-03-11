-- feature aggreagation
\timing on 
-- b) spatial feature aggregation (single tile)
SELECT 
    ST_Area(way) AS AREA,
    ST_Centroid(way) AS CENTROID,
    ST_ConvexHull(way) AS CONVHULL,
    ST_Perimeter(way) AS PERIMETER
FROM   	osm_polygon_planet
WHERE  	tilename ='497_763' ;



--d) spatial feature aggregation (collection)
SELECT 
    ST_Area(way) AS AREA,
    ST_Centroid(way) AS CENTROID,
    ST_ConvexHull(way) AS CONVHULL,
    ST_Perimeter(way) AS PERIMETER
FROM osm_polygon_planet ;


--e) spatial feature aggregation with spatial predicate (collection)
SELECT 
    ST_Area(way) AS AREA,
    ST_Centroid(way) AS CENTROID,
    ST_ConvexHull(way) AS CONVHULL,
    ST_Perimeter(way) AS PERIMETER
FROM osm_polygon_planet
WHERE ST_Area(way) > 0.5 ;

\timing off
