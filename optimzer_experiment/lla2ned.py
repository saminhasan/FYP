#lat_ref,  lon_ref, alt_ref = 23.862054, 90.361757, 0.0
import navpy as nv
nv.ned2lla(ned, lat_ref, lon_ref, alt_ref, latlon_unit='deg', alt_unit='m', model='wgs84')