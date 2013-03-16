-- c) two way join on a collection

SELECT ST_Area(ST_Intersection(A.polygon, B.polygon)) / ST_Area(ST_Union( A.polygon, B.polygon))  AS a_ratio,
    ST_DISTANCE( ST_Centroid(A.polygon), ST_Centroid(B.polygon) ) AS centroid_distance
FROM markup_polygon A, markup_polygon B
WHERE A.tilename = B.tilename AND A.seqnum = 1 AND B.seqnum = 2 AND ST_Intersects(A.polygon, B.polygon) = TRUE ;
