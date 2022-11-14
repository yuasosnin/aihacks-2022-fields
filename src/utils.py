import geojson
import shapely.geometry
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split


def parse_geometry(geom_str):
    try:
        return shapely.geometry.shape(geojson.loads(geom_str))
    except (TypeError, AttributeError):
        return None
        
def handle_bad_geometry(geometry):
    if isinstance(geometry, shapely.geometry.GeometryCollection):
        for geom in geometry.geoms:
            if isinstance(geom, shapely.geometry.Polygon):
                return geom
        else:
            return None
    else:
        return geometry
    
def read_data(filename):
    df = pd.read_csv(filename)
    df['geometry'] = df['.geo'].apply(parse_geometry).apply(handle_bad_geometry).copy()
    return gpd.GeoDataFrame(df, crs=4326)


def process_data(data):
    try:
        data_ts = data.drop(['id', 'area', '.geo', 'crop', 'geometry'], axis=1).copy()
    except KeyError:
        data_ts = data.drop(['id', 'area', '.geo', 'geometry'], axis=1).copy()
    data_ts.columns = pd.to_datetime(pd.to_datetime([x[8:] for x in data_ts.columns]))
    data_ts = data_ts[sorted(data_ts.columns)]
    try:
        data_id = data[['id', 'area', 'crop', 'geometry']].copy()
    except KeyError:
        data_id = data[['id', 'area', 'geometry']].copy()
        
    data_id['centroid'] = data_id['geometry'].to_crs('+proj=cea').centroid.to_crs(4326).copy()
    data_id['lon'] = data_id['centroid'].apply(lambda x: x.coords[0][0]).copy()
    data_id['lat'] = data_id['centroid'].apply(lambda x: x.coords[0][1]).copy()
    data_id = data_id.drop('centroid', axis=1).copy()
    
    return data_ts, data_id


def split_df(df, size=100):
    """https://stackoverflow.com/questions/54244560/split-pandas-dataframe-into-n-equal-parts-1"""
    n = int(len(df)/size)
    return [df.iloc[i*size:(i+1)*size].copy() for i in range(n+1)]
