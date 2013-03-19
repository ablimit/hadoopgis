SELECT current timestamp FROM sysibm.sysdummy1;
WITH PAISUID (A_PAIS_UID, B_PAIS_UID) AS (
		SELECT A.PAIS_UID, B.PAIS_UID
		FROM PAIS.collection A, pais.collection B
		where A.collection_uid = B.collection_uid
		and A.sequencenumber = 1
		and B.sequencenumber =2
		)
--select count (*) from (
		SELECT
		A.pais_uid
		,A.tilename
		--,DECIMAL(db2gse.ST_Area(a.polygon) / db2gse.ST_Area(b.polygon), 5, 2)
		-- ,DECIMAL(SQRT((a.centroid_x - b.centroid_x) * (a.centroid_x - b.centroid_x) + (a.centroid_y - b.centroi
		--			d_y) * (a.centroid_y - b.centroid_y)), 5, 2) AS xy_distance
		,CAST(  db2gse.ST_Area(db2gse.ST_Intersection(a.polygon, b.polygon) )/db2gse.ST_Area(db2gse.ST_UNION( a
					.polygon, b.polygon))  AS DECIMAL(4,2) ) AS area_ratio
		FROM pais.markup_polygon A, pais.markup_polygon B, paisuid P
		WHERE     A.tilename = B.tilename 
		and      P.A_PAIS_UID = a.pais_uid
		and      P.B_PAIS_UID = b.pais_uid
		and      DB2GSE.ST_intersects(A.polygon, B.polygon) = 1
		selectivity 0.000001 WITH UR ;

--		);

select current timestamp from sysibm.sysdummy1;
