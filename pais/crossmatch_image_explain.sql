-- single image 

EXPLAIN SELECT  A.pais_uid , A.tilename, ST_Distance(ST_Centroid(A.polygon), ST_Centroid(B.polygon)) AS centroid_distance, ST_Area(ST_Intersection(A.polygon, B.polygon) )/ST_Area(ST_Union( a.polygon, b.polygon)) AS area_ratio 

FROM markup_polygon AS A INNER JOIN markup_polygon B ON ST_intersects(A.polygon, B.polygon)
WHERE  A.tilename =B.tilename  
    AND      A.pais_uid = 'oligoIII.2_40x_20x_NS-MORPH_1' 
    AND      B.pais_uid = 'oligoIII.2_40x_20x_NS-MORPH_2' ;

