
-- Selection with field spatial predicate filtering  AVG(ST_AREA(polygon)) ==125.8
SELECT 	ST_AsText(polygon)
FROM	markup_polygon  
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND ST_Area(polygon) > 125.0 AND 
ST_Contains( ST_PolygonFromText('POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))'), polygon ) = TRUE ;


