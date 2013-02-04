
CREATE TABLE MARKUP_POLYGON(
       PAIS_UID		    VARCHAR(64) NOT NULL,
       TILENAME		VARCHAR(64) NOT NULL,
       MARKUP_ID	    DECIMAL(30,0) NOT NULL
) DISTRIBUTED BY (TILENAME); 

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
) DISTRIBUTED BY (NAME); 



