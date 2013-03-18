
EXPLAIN SELECT A.markup_id , B.markup_id
-- ST_Area(ST_Intersection(A.polygon, B.polygon)) / ST_Area(ST_Union( A.polygon, B.polygon))  AS a_ratio
FROM markup_polygon A JOIN markup_polygon B ON A.tilename = B.tilename AND A.seqnum = 1 AND B.seqnum = 2 ;


EXPLAIN SELECT A.markup_id , B.markup_id
-- ST_Area(ST_Intersection(A.polygon, B.polygon)) / ST_Area(ST_Union( A.polygon, B.polygon))  AS a_ratio
FROM markup_polygon A JOIN markup_polygon B ON ST_Intersects(A.polygon, B.polygon)  
WHERE A.tilename = B.tilename AND A.seqnum = 1 AND B.seqnum = 2 ;


EXPLAIN SELECT A.markup_id , B.markup_id
--ST_Area(ST_Intersection(A.polygon, B.polygon)) / ST_Area(ST_Union( A.polygon, B.polygon))  AS a_ratio
FROM markup_polygon A JOIN markup_polygon B ON A.polygon && B.polygon WHERE ST_Intersects(A.polygon, B.polygon)  
AND A.tilename = B.tilename AND A.seqnum = 1 AND B.seqnum = 2 ;

--EXPLAIN SELECT area_intersection / (area_a + area_b - area_intersection) AS intersection_ratio 
EXPLAIN SELECT *
FROM (  
    SELECT ST_Area(ST_Intersection(A.polygon, B.polygon)) AS area_intersection, ST_Area(A.polygon) AS area_a, ST_Area(B.polygon) AS area_b 
    FROM markup_polygon A , markup_polygon B WHERE A.polygon && B.polygon AND A.tilename=B.tilename AND A.seqnum=1 AND B.seqnum=2) AS tempjoin WHERE area_intersection > 0.0 ; 


EXPLAIN SELECT area_intersection / (area_a + area_b - area_intersection) AS intersection_ratio 
FROM (  
    SELECT ST_Area(ST_Intersection(A.polygon, B.polygon)) AS area_intersection, ST_Area(A.polygon) AS area_a, ST_Area(B.polygon) AS area_b 
    FROM markup_polygon A JOIN markup_polygon B  ON A.tilename=B.tilename WHERE A.seqnum=1 AND B.seqnum=2 AND A.polygon && B.polygon ) AS tempjoin WHERE area_intersection > 0.0 ; 

