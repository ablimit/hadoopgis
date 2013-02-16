-- Type 1: Containment Query

-- Given a human marked region (polygon), return boundaries of all nuclei polygons contained in this region.

-- a) Given a human-marked region in a single tile, which is defined by points (29500 35500, 34500 35500,34500 49000,29500 49000,29500 35500)


-- Selection with field filtering (small region == a single tile )
SELECT 	polygon
FROM   	pais.markup_polygon  
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND tilename ='gbm1.1-0000040960-0000040960' AND
ST_Contains( ST_PolygonFromText('POLYGON((40960 40960, 41984 40960,  41984 41984, 40960 41984, 40960 40960))'), polygon ) = TRUE ;


-- Selection without tile-field filtering (median region == a single image)
SELECT 	polygon
FROM   	pais.markup_polygon  
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND 
ST_Contains( ST_PolygonFromText('POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))'), polygon ) = TRUE ;

-- Selection without field filtering (large region ==  whole table == collection)
SELECT 	polygon
FROM   	pais.markup_polygon  
WHERE  ST_Contains( ST_PolygonFromText('POLYGON((40960 40960, 41984 40960,  41984 41984, 40960 41984, 40960 40960))'), polygon ) = TRUE ;



-- Selection with field spatial predicate filtering  AVG(ST_AREA(polygon)) ==125.8
SELECT 	polygon
FROM   	pais.markup_polygon  
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND ST_Area(polygon) > 125.0 AND 
ST_Contains( ST_PolygonFromText('POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))', 100), polygon ) = TRUE ;

