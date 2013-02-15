
SELECT  tilename, COUNT(*) AS load FROM osm_polygon_planet GROUP BY tilename ORDER BY load DESC ;

