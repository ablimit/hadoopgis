-- feature aggreagation

SELECT current timestamp FROM sysibm.sysdummy1;
SELECT 
AVG(DB2GSE.ST_Area(polygon)) AS AVG_AREA,
	--    ST_Centroid(polygon) AS CENTROID,
	--    ST_ConvexHull(polygon) AS CONVHULL,
	AVG(DB2GSE.ST_Perimeter(polygon)) AS AVG_PERIMETER
FROM   	PAIS.markup_polygon;
SELECT current timestamp FROM sysibm.sysdummy1;

--GROUP BY necrosis;

SELECT current timestamp FROM sysibm.sysdummy1;
SELECT 
AVG(DB2GSE.ST_Area(polygon)) AS AVG_AREA,
	--    ST_Centroid(polygon) AS CENTROID,
	--    ST_ConvexHull(polygon) AS CONVHULL,
	AVG(DB2GSE.ST_Perimeter(polygon)) AS AVG_PERIMETER,
	NECROSIS AS NECROSIS_LEVEL
FROM   	PAIS.markup_polygon GROUP BY NECROSIS;
SELECT current timestamp FROM sysibm.sysdummy1;

