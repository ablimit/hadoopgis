-- Selection without field filtering 
\timing on  

SELECT 	way
FROM   	osm_polygon_planet  
WHERE  	ST_Area(way) > 1.0 ;

