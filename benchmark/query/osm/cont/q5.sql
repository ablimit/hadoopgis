-- Selection without field filtering 

SELECT 	ST_AsText(way)
FROM   	osm_polygon_planet_fourxfour  
WHERE  	ST_Area(way) > 1.0 ;

