-- Find all intersected segmented nuclei (with intersection ratio and distance) between parameter set 1 and 2 of algorithm algorithm 'NS-MORPH' on tile 'oligoIII.2.ndpi-0000090112-0000024576':


-- single tile 

EXPLAIN 
    SELECT A.pais_uid, A.tilename, 
    (ST_Distance(ST_Centroid(b.polygon), ST_Centroid(a.polygon) )) AS centroid_distance, 
    (ST_Area(ST_Intersection(A.polygon, B.polygon))/ST_Area(ST_Union(A.polygon, B.polygon))) AS area_ratio
FROM markup_polygon AS A INNER JOIN markup_polygon AS B ON ST_Intersects(A.polygon, B.polygon)
WHERE  A.tilename ='gbm1.1-0000040960-0000040960' AND A.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_1' AND
       B.tilename ='gbm1.1-0000040960-0000040960' AND B.pais_uid = 'gbm1.1_40x_20x_NS-MORPH_2' ;

