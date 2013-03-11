-- Selection with field filtering (small region = tile)

SELECT 	ST_AsText(way) 
FROM   	osm_polygon_planet_fourxfour
WHERE  	tilename ='497_763' AND
ST_Contains( ST_PolygonFromText('POLYGON((-1.43999 47.16,-1.07999 47.16,-1.07999 47.34,-1.43999 47.34,-1.43999 47.16))', -1), way ) = TRUE ;
