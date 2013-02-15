-- Type 1: Containment Query

-- Selection with field filtering (small region)
SELECT 	way 
FROM   	osm_polygon_planet
WHERE  	tilename ='497_763' AND
ST_Contains( ST_PolygonFromText('POLYGON((-1.43999 47.16,-1.07999 47.16,-1.07999 47.34,-1.43999 47.34,-1.43999 47.16))', -1), way ) = TRUE ;

-- Selection without field filtering (small region)
SELECT 	way
FROM   	osm_polygon_planet 
WHERE  ST_Contains( ST_PolygonFromText('paris'), way) = TRUE ;


-- Selection with field filtering (large region)
SELECT 	way
FROM   	osm_polygon_planet 
WHERE  ST_Contains( ST_PolygonFromText('england'), way ) = TRUE ;

-- Selection with field filtering 
SELECT 	polygon
FROM   	osm_polygon_planet  
WHERE  	ST_Area(polygon) >  10 AND  ST_Contains( ST_PolygonFromText('england'), polygon ) = TRUE ;

-- Selection with field filtering 
SELECT 	polygon
FROM   	osm_polygon_planet  
WHERE  	ST_Area(way) > 10 ;

