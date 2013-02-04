-- Find all intersected segmented nuclei (with intersection ratio and distance) between parameter set 1 and 2 of algorithm algorithm 'NS-MORPH' on tile 'oligoIII.2.ndpi-0000090112-0000024576':

-- image level --
DROP TABLE jointable ;

CREATE TABLE jointableimage
AS (SELECT c.name, c.sequencenumber, t.polygon FROM markup_polygon AS t INNER JOIN collection AS c  ON (c.pais_uid =t.pais_uid) ) DISTRIBUTED BY (name); 


CREATE INDEX jointableimage_spidx ON jointableimage USING GIST ( polygon);

CREATE INDEX jointableimage_fidx ON jointableimage (name);

VACUUM VERBOSE ANALYZE jointableimage;


-- tile level 
CREATE TABLE jointabletile 
AS (SELECT t.pais_uid, t.tilename, c.sequencenumber, t.polygon FROM markup_polygon AS t INNER JOIN collection AS c  ON (c.pais_uid =t.pais_uid) ) DISTRIBUTED BY (tilename);

CREATE INDEX jointabletile_spidx ON jointabletile USING GIST (polygon);

CREATE INDEX jointabletile_fidx ON jointabletile (tilename);

VACUUM VERBOSE ANALYZE jointabletile;

