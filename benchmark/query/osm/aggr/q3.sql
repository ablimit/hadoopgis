-- feature aggreagation

--e) spatial feature aggregation with spatial predicate (collection)

SELECT tilename AS TID,
    AVG(ST_Area(way)) AS AVG_AREA,
--    ST_Centroid(way) AS CENTROID,
--    ST_ConvexHull(way) AS CONVHULL,
    AVG(ST_Perimeter(way)) AS AVG_PERIMETER
FROM osm_polygon_planet_fourxfour
WHERE ST_Area(way) > 0.1 GROUP BY tilename;

