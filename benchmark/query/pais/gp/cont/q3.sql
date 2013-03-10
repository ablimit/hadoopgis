
-- Selection without field filtering (large region ==  whole table == collection)
/* SELECT 	ST_AsText(polygon)
FROM   	markup_polygon  
WHERE  ST_Contains( ST_PolygonFromText('POLYGON((40960 40960, 41984 40960,  41984 41984, 40960 41984, 40960 40960))'), polygon ) = TRUE ;
*/

SELECT 	pais_uid,tilename,markup_id
FROM   	markup_polygon  
WHERE ST_Contains( ST_PolygonFromText('POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))'), polygon ) = TRUE ;
