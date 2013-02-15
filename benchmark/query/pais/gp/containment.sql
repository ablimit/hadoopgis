-- Type 1: Containment Query

-- Given a human marked region (polygon), return boundaries of all nuclei polygons contained in this region.

-- a) Given a human-marked region in a single tile, which is defined by points (29500 35500, 34500 35500,34500 49000,29500 49000,29500 35500)


-- Selection with field filtering (small region)
SELECT 	polygon
FROM   	pais.markup_polygon  
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND tilename ='gbm1.1-0000040960-0000040960' AND
ST_Contains( ST_PolygonFromText('POLYGON((40960 40960, 41984 40960,  41984 41984, 40960 41984, 40960 40960))', 100), polygon ) = TRUE ;

-- Selection without field filtering (small region)
SELECT 	polygon
FROM   	pais.markup_polygon  
WHERE  ST_Contains( ST_PolygonFromText('POLYGON((40960 40960, 41984 40960,  41984 41984, 40960 41984, 40960 40960))', 100), polygon ) = TRUE ;


-- Selection with field filtering (large region)
SELECT 	polygon
FROM   	pais.markup_polygon  
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND 
ST_Contains( ST_PolygonFromText('POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))', 100), polygon ) = TRUE ;

-- Selection with field filtering 
SELECT 	polygon
FROM   	pais.markup_polygon  
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND ST_Area(polygon) > 10 AND 
ST_Contains( ST_PolygonFromText('POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))', 100), polygon ) = TRUE ;

-- Selection with field filtering 
SELECT 	polygon
FROM   	pais.markup_polygon  
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND ST_Area(polygon) > 10 ;

-- b) Given a human-marked region in a single tile, which is stored in the MARKUP_POLYGON_HUMAN table.

-- SELECT 	p.polygon
-- FROM   		pais.markup_polygon p, pais.markup_polygon_human h  
-- WHERE  	p.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND  p.tilename ='gbm1.1-0000040960-0000040960' AND 
--	h.pais_uid = p.pais_uid AND h.tilename = p.tilename AND ST_Contains( h.polygon, p.polygon ) = TRUE ;


-- c) Given a human-marked region among multiple tiles, which is stored in the MARKUP_POLYGON_HUMAN table.


-- SELECT p.polygon
-- FROM 		pais.markup_polygon p, pais.markup_polygon_human h
-- WHERE 	p.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND p.pais_uid = h.pais_uid AND ST_Contains(h.polygon, p.polygon) = TRUE ;


