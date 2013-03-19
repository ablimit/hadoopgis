-- c) Given a human-marked region among multiple tiles, which is stored in the MARKUP_POLYGON_HUMAN table.

--SELECT p.polygon
--FROM 		pais.markup_polygon p, pais.markup_polygon_human h
--WHERE 	p.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' and 
--p.pais_uid = h.pais_uid and
--DB2GSE.ST_Contains(h.polygon, p.polygon) = 1
--selectivity 0.0625 WITH UR;

SELECT current timestamp FROM sysibm.sysdummy1;
SELECT count(*)	
FROM   	PAIS.markup_polygon  
WHERE DB2GSE.ST_Contains(DB2GSE.ST_PolyFromText('POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))',100), polygon ) = 1;
SELECT current timestamp FROM sysibm.sysdummy1;

--SELECT current timestamp FROM sysibm.sysdummy1;
--SELECT 	TILENAME AS TID ,MARKUP_ID AS MID ,DB2GSE.ST_AsText(POLYGON) AS SHAPE
--FROM   	PAIS.markup_polygon  
--WHERE DB2GSE.ST_Contains(DB2GSE.ST_PolyFromText('POLYGON((22528 8192,67584 8192,67584 24576,22528 24576,22528 8192))',100), polygon ) = 1;
--SELECT current timestamp FROM sysibm.sysdummy1;

