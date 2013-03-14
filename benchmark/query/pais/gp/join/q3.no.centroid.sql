-- c) two way join on a collection


SELECT ST_Area(ST_Intersection(A.polygon, B.polygon)) / ST_Area(ST_Union( A.polygon, B.polygon))  AS a_ratio,
FROM markup_polygon A JOIN markup_polygon B ON ST_Intersects(A.polygon, B.polygon)  
WHERE A.tilename = B.tilename AND A.seqnum = 1 AND B.seqnum = 2 ;
