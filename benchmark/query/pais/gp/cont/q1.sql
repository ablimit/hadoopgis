
-- Selection with field filtering (small region == a single tile )
SELECT 	ST_AsText(polygon) AS shape
FROM   	markup_polygon  
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND tilename ='gbm1.1-0000040960-0000040960' AND
ST_Contains( ST_PolygonFromText('POLYGON((40960 40960, 41984 40960,  41984 41984, 40960 41984, 40960 40960))'), polygon ) = TRUE ;
