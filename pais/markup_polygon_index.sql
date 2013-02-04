
CREATE INDEX markup_polygon_spidx ON MARKUP_POLYGON USING GIST ( polygon);

CREATE INDEX markup_polygon_fidx ON MARKUP_POLYGON (pais_uid,tilename);

VACUUM VERBOSE ANALYZE MARKUP_POLYGON;
