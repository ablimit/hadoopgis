-- Find all intersected segmented nuclei (with intersection ratio and distance) between parameter set 1 and 2 of algorithm algorithm 'NS-MORPH' on tile 'oligoIII.2.ndpi-0000090112-0000024576':

-- brute force all images --
\timing 

EXPLAIN
    SELECT  A.name, ST_Distance(ST_Centroid(A.polygon), ST_Centroid(B.polygon)) AS centroid_distance, 
    ST_Area(ST_Intersection(A.polygon, B.polygon))/ST_Area(ST_Union( a.polygon, b.polygon)) AS area_ratio 
    FROM jointableimage AS A INNER JOIN jointableimage AS B ON ST_intersects(A.polygon, B.polygon)
    WHERE  A.name =B.name  AND  A.sequencenumber = 1 AND B.sequencenumber = 2 ;

EXPLAIN 
    SELECT  A.tilename, ST_Distance(ST_Centroid(A.polygon), ST_Centroid(B.polygon)) AS centroid_distance, 
    ST_Area(ST_Intersection(A.polygon, B.polygon))/ST_Area(ST_Union( a.polygon, b.polygon)) AS area_ratio

    FROM jointabletile AS A INNER JOIN jointabletile AS B ON ST_intersects(A.polygon, B.polygon)
    WHERE  A.tilename = B.tilename  AND  A.sequencenumber = 1 AND B.sequencenumber = 2 ;


