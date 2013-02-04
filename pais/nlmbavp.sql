
-- Q. Find the nuclear density for each tile from algorithm 'NS-MORPH' with first parameter set:
insert into bchmk.querytime SELECT 'tiledensity', current timestamp, null  FROM sysibm.sysdummy1;
SELECT m.pais_uid, m.tilename, COUNT(*) AS count 
    FROM   pais.markup_polygon m,  pais.collection c
    WHERE  m.pais_uid=c.pais_uid AND c.methodname ='NS-MORPH' AND
       c.sequencenumber ='1'
    GROUP BY m.pais_uid, m.tilename;


---------------------------------------------------------------------
---------------------------------------------------------------------

-- Q. Retrieve nuclei boundaries with feature area between 200 and 500 and  eccentricity between 0 and 0.5 for tile 'gbm1.1-0000040960-0000040960':


SELECT CAST (m.POLYGON..ST_AsText AS  varchar(3000) ) AS POINTS  
FROM   pais.markup_polygon m, pais.calculation_flat c 
WHERE  m.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  AND m.tilename ='gbm1.1-0000040960-0000040960' AND
       c.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND  m.markup_id =  c.markup_id AND 
       c.area >= 200 AND c.area <= 500 AND 
       c.eccentricity >=0 and c.eccentricity <= 0.5;

update bchmk.querytime set endtime = current timestamp where queryname = 'tilefeatureselection';


---------------------------------------------------------------------
---------------------------------------------------------------------

-- Q. Retrieve nuclei boundaries with feature area between 200 and 500 and  eccentricity between 0 and 0.5 for slide 'gbm1.1:
echo 'slidefeatureselection';
insert into bchmk.querytime SELECT 'slidefeatureselection', current timestamp, null  FROM sysibm.sysdummy1;

SELECT pais.plgn2str(m.polygon) as boundary  
FROM   pais.markup_polygon m, pais.calculation_flat c 
WHERE  m.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND
       c.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND  m.markup_id =  c.markup_id AND 
       c.area >= 200 AND c.area <= 500 AND 
       c.eccentricity >=0 and c.eccentricity <= 0.5;
update bchmk.querytime set endtime = current timestamp where queryname = 'slidefeatureselection';

---------------------------------------------------------------------
---------------------------------------------------------------------
-- Q. Find all segmented nuclei boundaries of a tile 'gbm1.1-0000040960-0000040960':
echo 'boundariesforatile';
insert into bchmk.querytime SELECT 'boundariesforatile', current timestamp, null  FROM sysibm.sysdummy1;

SELECT markup_id, pais.plgn2str(polygon) as boundary   
FROM   pais.markup_polygon where pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND 
       tilename ='gbm1.1-0000040960-0000040960';

update bchmk.querytime set endtime = current timestamp where queryname = 'boundariesforatile';

---------------------------------------------------------------------
---------------------------------------------------------------------
-- Q. Given a human marked region from a viewer, return all nuclei boundaries
-- Retrieve all nuclei boundaries in a given region (29500 35500, 34500 35500,34500 49000,29500 49000,29500 35500):
echo 'viewerregion2boundaries';
insert into bchmk.querytime SELECT 'viewerregion2boundaries', current timestamp, null  FROM sysibm.sysdummy1;

SELECT pais.plgn2str(polygon) AS boundary
FROM   pais.markup_polygon  
WHERE  pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1'  and tilename ='gbm1.1-0000040960-0000040960' and 
DB2GSE.ST_Contains( DB2GSE.ST_Polygon('polygon((40960 40960, 41984 40960,  41984 41984, 40960 41984, 40960 40960))', 100),polygon ) = 1 
selectivity 0.0625 WITH UR; 

 update bchmk.querytime set endtime = current timestamp where queryname = 'viewerregion2boundaries';

---------------------------------------------------------------------
---------------------------------------------------------------------
-- Given a point marked by a human in a necleus, return the features of this necleus:
echo 'point2features';
insert into bchmk.querytime SELECT 'point2features', current timestamp, null  FROM sysibm.sysdummy1;

SELECT c.area, c.perimeter, c.eccentricity
FROM   PAIS.CALCULATION_FLAT c, PAIS.MARKUP_POLYGON m
WHERE  m.PAIS_UID = 'astroII.1_40x_20x_NS-MORPH_1' AND c.PAIS_UID = m.PAIS_UID and 
       C.markup_id=M.markup_id AND 
       DB2GSE.ST_Intersects(M.polygon, DB2GSE.ST_Point(9342, 7316, 100))=1  selectivity   0.001 with UR;

update bchmk.querytime set endtime = current timestamp where queryname = 'point2features';

---------------------------------------------------------------------
---------------------------------------------------------------------
-- Find all intersected segmented nuclei (with intersection ratio and distance) between parameter set 1 and 2 of algorithm algorithm 'NS-MORPH' on tile 'oligoIII.2.ndpi-0000090112-0000024576':

echo 'intersectednucleiforatile';
insert into bchmk.querytime SELECT 'intersectednucleiforatile', current timestamp, null  FROM sysibm.sysdummy1;

select count(*) from (
SELECT A.pais_uid, A.tilename,  CAST(  db2gse.ST_Area(db2gse.ST_Intersection(a.polygon, b.polygon) )/db2gse.ST_Area(db2gse.ST_UNION( a.polygon, b.polygon))  AS DECIMAL(4,2) )
AS area_ratio,  CAST( DB2GSE.ST_DISTANCE( db2gse.ST_Centroid(b.polygon), db2gse.ST_Centroid(a.polygon) )  AS DECIMAL(5,2) ) AS centroid_distance
FROM pais.markup_polygon A, pais.markup_polygon B
WHERE  A.tilename ='gbm1.1-0000040960-0000040960' and A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' and 
       B.tilename ='gbm1.1-0000040960-0000040960' and B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' and
       DB2GSE.ST_intersects(A.polygon, B.polygon) = 1 selectivity 0.0001 WITH UR
); 

update bchmk.querytime set endtime = current timestamp where queryname = 'intersectednucleiforatile';

---------------------------------------------------------------------
---------------------------------------------------------------------
-- Difference or missed nuclei for one algorithm
echo 'noneintersectnucleiinatile';
insert into bchmk.querytime SELECT 'noneintersectnucleiinatile', current timestamp, null  FROM sysibm.sysdummy1;

SELECT count(*) FROM pais.markup_polygon
WHERE tilename ='gbm1.1-0000040960-0000040960' and pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' AND
markup_id NOT IN 
(SELECT B.markup_id
FROM pais.markup_polygon A, pais.markup_polygon B
WHERE  A.tilename = 'gbm1.1-0000040960-0000040960' and A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' and 
       B.tilename = 'gbm1.1-0000040960-0000040960' and B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' and
       DB2GSE.ST_intersects(A.polygon, B.polygon) = 1 selectivity 0.0001 WITH UR
);


update bchmk.querytime set endtime = current timestamp where queryname = 'noneintersectnucleiinatile';

---------------------------------------------------------------------
---------------------------------------------------------------------
-- Find how many markups in Set A with multiple intersects in Set B:
echo 'multiintersectatile';
insert into bchmk.querytime SELECT 'multiintersectatile', current timestamp, null  FROM sysibm.sysdummy1;

SELECT  A.pais_uid, A.tilename,  A.markup_id, count(*)
FROM pais.markup_polygon A, pais.markup_polygon B
WHERE  A.tilename ='gbm1.1-0000040960-0000040960' and A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' and 
       B.tilename ='gbm1.1-0000040960-0000040960' and B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' and
       DB2GSE.ST_intersects(A.polygon, B.polygon) = 1 selectivity 0.0001 
GROUP BY (A.pais_uid, A.tilename, A.markup_id) HAVING COUNT(*) > 1 WITH UR;

update bchmk.querytime set endtime = current timestamp where queryname = 'multiintersectatile';

---------------------------------------------------------------------
---------------------------------------------------------------------
-- For each human marked region type, count the number of nuclei in that region and the average and STDDEV of the classification value:
echo 'humanregionavggrade';
insert into bchmk.querytime SELECT 'humanregionavggrade', current timestamp, null  FROM sysibm.sysdummy1;

SELECT n.QUANTIFICATION_VALUE, count(m.markup_id) NUM_NUCLEI, 
       CAST(avg (o.QUANTIFICATION_VALUE) AS  DECIMAL(2,1) ) AVG_GRADE,
       CAST(stddev (o.QUANTIFICATION_VALUE) AS  DECIMAL(5,3) ) STD_GRADE
FROM   PAIS.MARKUP_POLYGON m, PAIS.MARKUP_POLYGON_HUMAN r, PAIS.OBSERVATION_QUANTIFICATION_NOMINAL n,
       PAIS.OBSERVATION_QUANTIFICATION_ORDINAL o
WHERE  m.pais_uid ='gbm1.1_40x_20x_NS-MORPH_1' AND o.pais_uid = m.pais_uid AND 
       r.pais_uid = 'gbm1.1_40x_40x_RG-HUMAN_1' AND n.pais_uid = r.pais_uid AND
       o.markup_id = m.markup_id and  n.markup_id = r.markup_id AND
       n.OBSERVATION_NAME ='Diffuse Glioma Classification' AND
       DB2GSE.ST_Contains( r.polygon, m.polygon) = 1
GROUP BY (n.QUANTIFICATION_VALUE);

update bchmk.querytime set endtime = current timestamp where queryname = 'humanregionavggrade';


---------------------------------------------------------------------
---------------------------------------------------------------------
-- For each human marked region type, count the number of nuclei in that region and the average and STDDEV of the classification value:
(=
SELECT n.QUANTIFICATION_VALUE, count(m.markup_id) NUM_NUCLEI, 
       CAST(avg (o.QUANTIFICATION_VALUE) AS  DECIMAL(2,1) ) AVG_GRADE,
       CAST(stddev (o.QUANTIFICATION_VALUE) AS  DECIMAL(5,3) ) STD_GRADE
FROM   PAIS.MARKUP_POLYGON m, PAIS.MARKUP_POLYGON_HUMAN r, PAIS.OBSERVATION_QUANTIFICATION_NOMINAL n,
       PAIS.OBSERVATION_QUANTIFICATION_ORDINAL o, PAIS.COLLECTION c1, PAIS.COLLECTION c2
WHERE  c1.name = c2.name and c1.role='algorithm' and c1.sequencenumber ='1' AND
       c1.pais_uid = m.pais_uid and c2.pais_uid = r.pais_uid AND
       o.pais_uid = m.pais_uid AND  n.pais_uid = r.pais_uid AND
       o.markup_id = m.markup_id and  n.markup_id = r.markup_id AND
       n.OBSERVATION_NAME ='Diffuse Glioma Classification' AND
       DB2GSE.ST_Contains( r.polygon, m.polygon) = 1
GROUP BY (n.QUANTIFICATION_VALUE);
=)



