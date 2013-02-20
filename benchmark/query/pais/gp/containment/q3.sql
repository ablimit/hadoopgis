
-- Selection without field filtering (large region ==  whole table == collection)
SELECT 	ST_AsText(polygon)
FROM   	markup_polygon  
WHERE  ST_Contains( ST_PolygonFromText('POLYGON((40960 40960, 41984 40960,  41984 41984, 40960 41984, 40960 40960))'), polygon ) = TRUE ;


