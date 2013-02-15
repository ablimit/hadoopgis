
SELECT gp_segment_id, count(*) FROM osm_polygon_planet GROUP BY 1 ORDER BY 2 DESC;
