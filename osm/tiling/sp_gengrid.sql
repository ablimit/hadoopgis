CREATE OR REPLACE FUNCTION gengrid (_lat integer, _lon integer, tname varchar) RETURNS integer AS $$

DECLARE 
v_width  real := 360.0;
v_height real := 180.0;
v_ox  real := -180.0;
v_oy  real := -90.0;

v_x0 real := 0.0;
v_y0 real := 0.0;
v_x1 real := 0.0;
v_y1 real := 0.0;

xstep real := 0.0;
ystep real := 0.0;

i integer := 0 ;
j integer := 0 ;

tilename varchar := '' ;
v_polygonstr varchar := '' ;

BEGIN

IF EXISTS (SELECT 0 FROM pg_class where relname = quote_ident(tname) )
    THEN
      RAISE EXCEPTION 'Table % already exists!', tname ;
    RETURN 0 ;
END IF;



EXECUTE 'CREATE TABLE  '|| quote_ident(tname) ||' (
    tilename varchar(64) PRIMARY KEY,
    x       real,
    y       real,
    width   real,
    height  real
)';

EXECUTE ' SELECT AddGeometryColumn(' || quote_literal(tname) || ',' || quote_literal('mbb') || ',' || '4326, ' || 
    quote_literal('POLYGON') || ', 2 )' ;


xstep = v_width/_lat ;
ystep = v_height/_lon ;

FOR i IN 1.._lat LOOP
    FOR j IN 1.._lon LOOP
	-- CALL DBMS_OUTPUT.PUT_LINE('i: ' || i || ' j: ' || j);

	tilename :=  i || '_' || j;
	v_x0 := v_ox + xstep * (i-1) ;
	v_y0 := v_oy + ystep * (j-1) ;
	v_x1 := v_ox + xstep * i;
	v_y1 := v_oy + ystep * j;

	v_polygonstr := 'POLYGON((' ||  v_x0 || ' '|| v_y0 ||',' || v_x1 || ' '|| v_y0 ||',' || v_x1 || ' '|| v_y1
		||',' || v_x0 || ' '|| v_y1 ||',' || v_x0 || ' '|| v_y0 ||'))';


	EXECUTE 'INSERT INTO ' || quote_ident(tname) || '( tilename, x, y, width, height, mbb) VALUES(' || 
	    quote_literal(tilename) || ',' || v_x0 || ',' || v_y0 || ',' || xstep || ',' || ystep || ', ST_PolygonFromText(' || quote_literal(v_polygonstr) ||', 4326))' ;

    END LOOP;
END LOOP;

RETURN 1;

END;

$$ LANGUAGE  'plpgsql'
