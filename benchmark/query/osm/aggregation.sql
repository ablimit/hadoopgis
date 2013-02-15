-- feature aggreagation

-- For each human marked region (polygon), count the number of nuclei in that region and the average features of them.

-- a) flat feature aggregation w selection 
SELECT 	count(m.markup_id) NUM_NUCLEI, 
    avg(f.AREA) AVG_AREA,
    avg(f.eccentricity) AVG_ECC,
    avg(f.PERIMETER) AVG_PERIMETER
FROM   	osm_polygon_planet
WHERE  	


-- b) spatial feature aggregation w/o selection
SELECT 	count(polygon) NUM_NUCLEI, 
    avg(f.AREA) AVG_AREA,
    avg(f.eccentricity) AVG_ECC,
    avg(f.PERIMETER) AVG_PERIMETER
FROM   	pais.markup_polygon  
WHERE  	pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND ST_Area(polygon) > 10 AND 
ST_Contains( ST_PolygonFromText('POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))', 100), polygon ) = TRUE ;

