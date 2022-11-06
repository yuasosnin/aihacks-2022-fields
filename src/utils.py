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
        
    data_id['centroid'] = data_id['geometry'].to_crs(4327).centroid.copy()
    data_id['lat'] = data_id['centroid'].apply(lambda x: x.coords[0][0]).copy()
    data_id['lon'] = data_id['centroid'].apply(lambda x: x.coords[0][1]).copy()
    data_id = data_id.drop('centroid', axis=1).copy()
    
    return data_ts, data_id


def ndvi(red, nir):
    return (nir-red)/(nir+red)


def split_df(df, size=100):
    """https://stackoverflow.com/questions/54244560/split-pandas-dataframe-into-n-equal-parts-1"""
    n = int(len(df)/size)
    return [df.iloc[i*size:(i+1)*size].copy() for i in range(n+1)]


def get_dataset(data_ts, data_id, data_ts_pred, data_id_pred, scale=True, random_state=1):
    data_ts_train, data_ts_val, data_id_train, data_id_val = train_test_split(data_ts, data_id, test_size=0.2, random_state=random_state)
    data_ts_val, data_ts_test, data_id_val, data_id_test = train_test_split(data_ts_val, data_id_val, test_size=0.5, random_state=random_state)

    data_train = data_ts_train
    data_val = data_ts_val
    data_test = data_ts_test
    data_pred = data_ts_pred
    
    if scale:
        mean = data_train.values.mean()
        std = data_train.values.std()
    else:
        mean = 0
        std = 1
    
    dataset = dict()
    dataset['full'] = {'X': (data_ts-mean)/std, 'y': data_id['crop']}
    dataset['train'] = {'X': (data_train-mean)/std, 'y': data_id_train['crop']}
    dataset['val'] = {'X': (data_val-mean)/std, 'y': data_id_val['crop']}
    dataset['test'] = {'X': (data_test-mean)/std, 'y': data_id_test['crop']}
    dataset['pred'] = {'X': (data_pred-mean)/std}
    return dataset
