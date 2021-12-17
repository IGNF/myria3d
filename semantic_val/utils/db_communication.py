import subprocess

def db_communication(
	xmin: float, 
	ymin: float, 
	xmax: float, 
	ymax: float,
	shapefile_path: str,
	):
	
	host = "serveurbdudiff.ign.fr"
	user = "invite"
	pwd = "28de#"
	bd_name = "bduni_france_consultation"
	srid = 2154
	
	sql_request = f"SELECT st_setsrid(batiment.geometrie,{srid}) AS geometry, nature FROM batiment WHERE batiment.geometrie && ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, {srid}) and not gcms_detruit"
	cmd = ["pgsql2shp", "-f", shapefile_path, "-h", host, "-u", user, "-P", pwd, bd_name, sql_request]
	subprocess.call(cmd)