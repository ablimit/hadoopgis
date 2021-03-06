
BEGIN;
CREATE EXTENSION postgis;

CREATE TABLE MARKUP_POLYGON(
       PAIS_UID		    VARCHAR(64) NOT NULL,
       TILENAME		VARCHAR(64) NOT NULL,
       MARKUP_ID	    DECIMAL(30,0) NOT NULL
) ;

SELECT AddGeometryColumn('public','markup_polygon', 'polygon', -1, 'POLYGON', 2);

CREATE TABLE COLLECTION(  
      COLLECTION_ID	    DECIMAL(30,0) NOT NULL,
      COLLECTION_UID	    VARCHAR(64) NOT NULL,
      NAME		VARCHAR(64),
      ROLE		VARCHAR(64),
      METHODNAME	    VARCHAR(64) NOT NULL,
      SEQUENCENUMBER	    INTEGER NOT NULL,
      STUDYDATETIME		VARCHAR(64),
      PAIS_UID		VARCHAR(64) NOT NULL 
);

COPY markup_polygon FROM '/data2/ablimit/db2exp/markup_polygon.sql' WITH DELIMITER '|' CSV HEADER ;

COPY collection FROM '/data2/ablimit/db2exp/collection.sql' WITH DELIMITER '|';

CREATE INDEX markup_polygon_spidx ON MARKUP_POLYGON USING GIST ( polygon);

CREATE INDEX markup_polygon_fidx ON MARKUP_POLYGON (pais_uid,tilename);

END;

VACUUM VERBOSE ANALYZE MARKUP_POLYGON;
