
BEGIN;

CREATE TABLE SIXALGO (
	PAIS_UID		    VARCHAR(64) NOT NULL,
	TILENAME		VARCHAR(64) NOT NULL,
	SEQUENCENUMBER INTEGER NOT NULL
) DISTRIBUTED BY (TILENAME); 

SELECT AddGeometryColumn('public','sixalgo', 'polygon', -1, 'POLYGON', 2);

COPY sixalgo FROM '/data2/ablimit/Data/spatialdata/pais/sixalgo.dat' WITH HEADER DELIMITER '|' ;


COMMIT ;


CREATE INDEX sixalgo_sp_idx ON sixalgo USING GIST (polygon);

CREATE INDEX sixalgo_fidx ON sixalgo (tilename);

VACUUM VERBOSE ANALYZE sixalgo;


