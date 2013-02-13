
-- Type 1: Containment Query

-- Given a human marked region (polygon), return boundaries of all nuclei polygons contained in this region.

-- a) Given a human-marked region in a single tile, which is defined by points (29500 35500, 34500 35500,34500 49000,29500 49000,29500 35500)

SELECT 	polygon
FROM   	pais.markup_polygon  
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND tilename ='gbm1.1-0000040960-0000040960' AND
ST_Contains( ST_PolygonFromText('POLYGON((40960 40960, 41984 40960,  41984 41984, 40960 41984, 40960 40960))', 100), polygon ) = TRUE ;

-- b) Given a human-marked region in a single tile, which is stored in the MARKUP_POLYGON_HUMAN table.

SELECT 	p.polygon
FROM   		pais.markup_polygon p, pais.markup_polygon_human h  
WHERE  	p.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND  p.tilename ='gbm1.1-0000040960-0000040960' AND 
	h.pais_uid = p.pais_uid AND h.tilename = p.tilename AND ST_Contains( h.polygon, p.polygon ) = TRUE ;


-- c) Given a human-marked region among multiple tiles, which is stored in the MARKUP_POLYGON_HUMAN table.


SELECT p.polygon
FROM 		pais.markup_polygon p, pais.markup_polygon_human h
WHERE 	p.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND p.pais_uid = h.pais_uid AND ST_Contains(h.polygon, p.polygon) = TRUE ;

--d) SELECT cells in a small region 
--e) SELECT cells in a big region.


