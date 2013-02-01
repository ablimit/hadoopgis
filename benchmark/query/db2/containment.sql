
-- Type 1: Containment Query

-- Given a human marked region (polygon), return boundaries of all nuclei polygons contained in this region.

-- a) Given a human-marked region in a single tile, which is defined by points (29500 35500, 34500 35500,34500 49000,29500 49000,29500 35500)

SELECT 	polygon
FROM   	pais.markup_polygon  
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND tilename ='gbm1.1-0000040960-0000040960' AND
DB2GSE.ST_Contains( DB2GSE.ST_Polygon('polygon((40960 40960, 41984 40960,  41984 41984, 40960 41984, 40960 40960))', 100),polygon ) = 1 
selectivity 0.0625 WITH UR; 


-- b) Given a human-marked region in a single tile, which is stored in the MARKUP_POLYGON_HUMAN table.

SELECT 	p.polygon
FROM   		pais.markup_polygon p, pais.markup_polygon_human h  
WHERE  	p.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  and 
p.tilename ='gbm1.1-0000040960-0000040960' and 
	h.pais_uid = p.pais_uid and h.tilename = p.tilename and
	DB2GSE.ST_Contains( h.polygon, p.polygon ) = 1 
selectivity 0.0625 WITH UR; 


-- c) Given a human-marked region among multiple tiles, which is stored in the MARKUP_POLYGON_HUMAN table.


SELECT p.polygon
FROM 		pais.markup_polygon p, pais.markup_polygon_human h
WHERE 	p.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' and 
p.pais_uid = h.pais_uid and
DB2GSE.ST_Contains(h.polygon, p.polygon) = 1
selectivity 0.0625 WITH UR;


