# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import pandas as pd
import numpy as np
import requests
from sumup_lib import *


# loading data
try:
    df_sumup = pd.read_csv('data/SUMup 2022/SUMup_temperature_2022.csv')
except:
    print('Downloading SUMup 2022 temperature file')
    url = 'https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A9878109d-7b51-4dd5-a94c-765e44945e46'
    r = requests.get(url, allow_redirects=True)
    open('data/SUMup 2022/SUMup_temperature_2022.csv', 'wb').write(r.content)    
    df_sumup = pd.read_csv('data/SUMup 2022/SUMup_temperature_2022.csv')

df_sumup.columns = df_sumup.columns.str.lower()
df_sumup = df_sumup.rename(columns={'citation': 'reference'})

# correcting an error in SUMup 2022:
df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'longitude'] = -df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'longitude']


df_ref = pd.read_csv('data/SUMup 2022/SUMup_temperature_references_2022.txt', sep='eirj', engine='python', header=None)
df_ref.columns = ['reference']
df_ref['key'] = np.arange(1,len(df_ref)+1)
df_ref = df_ref.set_index('key')

df_names = pd.read_csv('data/SUMup 2022/SUMup_temperature_names_2022.txt', sep='eirj', engine='python', header=None)
df_names.columns = ['name']
df_names['key'] = np.arange(1,len(df_names)+1)
df_names = df_names.set_index('key')

df_methods = pd.read_csv('data/SUMup 2022/SUMup_temperature_methods_2022.txt',sep='\t')
df_methods = df_methods.set_index('key')

df_sumup['name_str'] = df_names.loc[df_sumup.name, 'name'].values
df_sumup['reference_full'] = df_ref.loc[df_sumup.reference, 'reference'].values
df_sumup['method_str'] = df_methods.loc[df_sumup.method, 'method'].values
short_ref = parse_short_reference(df_sumup).set_index('reference')
df_sumup['reference_short'] = short_ref.loc[df_sumup.reference_full].reference_short.values

df_sumup['notes'] = ''
len_sumup_2022 = df_sumup.shape[0]

print(df_sumup.shape[0], 'temperature observations currently in SUMup from', len(df_sumup.reference_full.unique()), 'sources')
print(df_sumup.loc[df_sumup.latitude>0].shape[0], 'in Greenland')
print(df_sumup.loc[df_sumup.latitude<0].shape[0], 'in Antarctica')


# %% Adding Vandecrux data
print('## Vandecrux et al. compilation')
df_vdx = pd.read_csv('data/temperature data/Vandecrux temperature/10m_temperature_dataset_monthly.csv')
df_vdx = df_vdx.rename(columns={'site':'name_str', 
                                'depthOfTemperatureObservation': 'depth',
                                'temperatureObserved': 'temperature',
                                'durationOpen': 'open_time',
                                'durationMeasured': 'duration',
                                'method': 'method_str',
                                'reference': 'reference_full',
                                'date':'timestamp',
                                'note':'notes'})
df_vdx['date'] = df_vdx.timestamp.str.replace('-','').astype(int)

print(df_sumup.shape[0], 'observations currently in SUMup')
print(df_vdx.shape[0], 'observations in new dataset')

print('Checking conflicts:\n')

sumup_index_conflict = check_conflicts(df_sumup, df_vdx, var=['name', 'depth','temperature'])

print('\noverwriting conflicting data in SUMup (checked by bav)\n')

msk = ~df_sumup.index.isin(sumup_index_conflict)
df_sumup = pd.concat((df_sumup.loc[msk,:], df_vdx), ignore_index=True)

#%% Amory compilation
print('loading Amory compilation')
df=pd.read_csv('data/temperature data/Amory compilation/T10m_Amory.csv', sep=';')
df.columns =['name_str','latitude','longitude','depth','temperature','reference_short', 
             'date','elevation','reference_full','method_str']
df['date'] = df.date.str.replace('//','/')
dt_start = pd.to_datetime(df.date.str.split('-').str[0], errors='coerce')
df['timestamp'] = dt_start.dt.strftime('%Y-%m-%d')
dt_end = pd.to_datetime(df.date.str.split('-').str[1], errors='coerce')
msk = df.timestamp.isnull()
df.loc[msk, 'timestamp'] = dt_end.loc[msk].dt.strftime('%Y-%m-%d')
df['duration'] = dt_end - dt_start
df['notes'] = df.method_str.str.split(',').str[1]
df['method_str'] = df.method_str.str.split(',').str[0]
df.loc[34, 'notes'] = df.loc[34, 'method_str']
df.loc[34, 'method_str'] = np.nan
df['latitude'] = df.latitude.str.replace('_','-')
df['reference_short'] = df['reference_short'].str.replace(r'([A-Za-z]+)(\d{4})', lambda m: m.group(1).capitalize() + ' et al. (' + m.group(2) + ')', regex=True)
df = df.loc[df.timestamp.notnull(),:] 
df['temperature'] = (df.temperature
                     .str.replace('Ð','-')
                     .str.replace('Ê','-')
                     .str.replace('--','-')
                     )
df['latitude'] =pd.to_numeric(df.latitude)
df['temperature'] =pd.to_numeric(df.temperature)
df_sumup = pd.concat((df_sumup, df), ignore_index=True)



# %% Removing duplicate obs, reference and renaming method
# renaming some redundant references
df_sumup.loc[df_sumup.method_str=='thermistor', 'method_str'] = 'Thermistor'
df_sumup.loc[df_sumup.method_str=='thermistors', 'method_str'] = 'Thermistor'
df_sumup.loc[df_sumup.method_str=='thermistor string', 'method_str'] = 'Thermistor'
df_sumup.loc[df_sumup.method_str=='custom thermistors', 'method_str'] = 'Thermistor'
df_sumup.loc[df_sumup.method_str.isnull(), 'method_str'] = 'not available'
df_sumup.loc[df_sumup.method_str=='not_reported', 'method_str'] = 'not available'
df_sumup.loc[df_sumup.method_str=='digital Thermarray system from RST©', 'method_str'] = 'RST ThermArray'
df_sumup.loc[df_sumup.method_str=='digital thermarray system from RST©', 'method_str'] = 'RST ThermArray'

# looking for redundant references
df_sumup.loc[df_sumup.reference_full.str.startswith('Miller'),'reference_short'] = 'Miller et al. (2020)'
df_sumup.loc[df_sumup.reference_full.str.startswith('Miller'),'reference_full'] = 'Miller, O., Solomon, D.K., Miège, C., Koenig, L., Forster, R., Schmerr, N., Ligtenberg, S.R., Legchenko, A., Voss, C.I., Montgomery, L. and McConnell, J.R., 2020. Hydrology of a perennial firn aquifer in Southeast Greenland: an overview driven by field data. Water Resources Research, 56(8), p.e2019WR026348. Dataset doi:10.18739/A2R785P5W'

df_sumup.loc[df_sumup.reference_full.str.startswith('Fausto'), 'reference_short'] = 'PROMICE/GC-Net, How et al. (2023)'
df_sumup.loc[df_sumup.reference_full.str.startswith('Fausto'), 'reference_full'] =  'Fausto, R. S., van As, D., Mankoff, K. D., Vandecrux, B., Citterio, M., Ahlstrøm, A. P., Andersen, S. B., Colgan, W., Karlsson, N. B., Kjeldsen, K. K., Korsgaard, N. J., Larsen, S. H., Nielsen, S., Pedersen, A. Ø., Shields, C. L., Solgaard, A. M., and Box, J. E.: Programme for Monitoring of the Greenland Ice Sheet (PROMICE) automatic weather station data, Earth Syst. Sci. Data, 13, 3819–3845, https://doi.org/10.5194/essd-13-3819-2021 , 2021. and How, P., Ahlstrøm, A.P., Andersen, S.B., Box, J.E., Citterio, M., Colgan, W.T., Fausto, R., Karlsson, N.B., Jakobsen, J., Larsen, S.H., Mankoff, K.D., Pedersen, A.Ø., Rutishauser, A., Shields, C.L., Solgaard, A.M., van As, D., Vandecrux, B., Wright, P.J., PROMICE and GC-Net automated weather station data in Greenland, https://doi.org/10.22008/FK2/IW73UU, GEUS Dataverse, 2022.'

df_sumup.loc[df_sumup.reference_full.str.startswith('Clausen HB and Stauffer B (1988)'),
             'reference_short'] = 'Clausen and Stauffer (1988)'

df_sumup.loc[df_sumup.reference_full.str.startswith('Charalampidis'),
             'reference_short'] = 'Charalampidis et al. (2016, 2022)'
# tmp = df_sumup.reference_full.unique()

print(df_sumup.shape[0], 'temperature observations after merging from', len(df_sumup.reference_full.unique()), 'sources')
print(df_sumup.loc[df_sumup.latitude>0].shape[0], 'in Greenland')
print(df_sumup.loc[df_sumup.latitude<0].shape[0], 'in Antarctica')

# %% writin CSV files
df_ref_new = df_sumup[['reference_full', 'reference_short']].drop_duplicates()
df_ref_new.columns = ['reference', 'reference_short']
df_ref_new['key'] = np.arange(1,len(df_ref_new)+1)
df_ref_new = df_ref_new.set_index('key')
df_ref_new.to_csv('SUMup 2023 beta/SUMup_2023_temperature_csv/SUMup_2023_temperature_references.tsv', sep='\t')
df_sumup['reference'] = df_ref_new.reset_index().set_index('reference').loc[df_sumup.reference_full, 'key'].values

df_method_new = pd.DataFrame(df_sumup.method_str.unique())
df_method_new.columns = ['method']
df_method_new['key'] = np.insert( np.arange(1,len(df_method_new)), 0, -9999)
df_method_new = df_method_new.set_index('key')
df_method_new.to_csv('SUMup 2023 beta/SUMup_2023_temperature_csv/SUMup_2023_temperature_methods.tsv', sep='\t')
df_sumup['method'] = df_method_new.reset_index().set_index('method').loc[df_sumup.method_str].values

df_name_new = pd.DataFrame(df_sumup.name_str.unique())
df_name_new.columns = ['name']
df_name_new['key'] = np.arange(1,len(df_name_new)+1)
df_name_new = df_name_new.set_index('key')
df_name_new.to_csv('SUMup 2023 beta/SUMup_2023_temperature_csv/SUMup_2023_temperature_names.tsv', sep='\t')
df_sumup['name'] = df_name_new.reset_index().set_index('name').loc[df_sumup.name_str].values

df_sumup['notes'] = ''
for var_int in ['elevation','open_time', 'duration']:
    df_sumup.loc[df_sumup[var_int].isnull(), var_int] = -9999
    df_sumup.loc[df_sumup[var_int]=='', var_int] = -9999
    # if there was something that is not a number, then we shift it to 'notes'
    df_sumup.loc[pd.to_numeric(df_sumup[var_int], errors='coerce').isnull(), 'notes'] = \
        df_sumup.loc[pd.to_numeric(df_sumup[var_int], errors='coerce').isnull(), var_int]
    df_sumup.loc[pd.to_numeric(df_sumup[var_int], errors='coerce').isnull(), var_int] = -9999
    df_sumup[var_int] = pd.to_numeric(df_sumup[var_int], errors='coerce').round(0).astype(int)
    df_sumup[var_int] = df_sumup[var_int].astype(str).replace('-9999','')

df_sumup['error'] = pd.to_numeric(df_sumup.error, errors='coerce')
df_sumup.loc[df_sumup.error == -9999, 'error'] = np.nan

df_sumup['timestamp'] = pd.to_datetime(df_sumup.timestamp)
df_sumup.latitude = df_sumup.latitude.round(6)
df_sumup.longitude = df_sumup.longitude.round(6)
df_sumup.temperature = df_sumup.temperature.round(3)

df_sumup = df_sumup.rename(columns={'name':'name_key',
                                'name_str':'name',
                                'method':'method_key',
                                'method_str':'method',
                                'reference':'reference_key',
                                'reference_full':'reference'})

var_to_csv = ['name_key', 'reference_key', 'method_key', 'timestamp', 'latitude',
        'longitude', 'elevation', 'depth', 'open_time', 'duration',
        'temperature', 'error']
df_sumup.loc[df_sumup.latitude>0, var_to_csv].to_csv('SUMup 2023 beta/SUMup_2023_temperature_csv/SUMup_2023_temperature_greenland.csv',index=None)

df_sumup.loc[df_sumup.latitude<0, var_to_csv].to_csv('SUMup 2023 beta/SUMup_2023_temperature_csv/SUMup_2023_temperature_antarctica.csv',index=None)


import shutil
shutil.make_archive('SUMup 2023 beta/SUMup_2023_temperature_csv',
                    'zip', 'SUMup 2023 beta/SUMup_2023_temperature_csv')

# %% netcdf format
import xarray as xr
df_sumup[['elevation','open_time', 'duration']] = \
    df_sumup[['elevation','open_time', 'duration']].replace('','-9999').astype(int)

def write_netcdf(df_sumup, filename):
    df_new = df_sumup.copy()
    df_new['timestamp'] = pd.to_datetime(df_new.timestamp).dt.tz_localize(None)

    df_new.index.name='measurement_id'
    assert (~df_new.index.duplicated()).all(), 'non-unique measurement-id "'

    ds_meta_name = (df_new[['name_key','name']]
                    .drop_duplicates()
                    .set_index('name_key')
                    .sort_index()
                    .to_xarray())
    ds_meta_method = (df_new[['method_key','method']]
                      .drop_duplicates()
                      .set_index('method_key')
                      .sort_index()
                      .to_xarray())
    ds_reference_method = (df_new[['reference_key', 'reference', 'reference_short']]
                           .drop_duplicates()
                           .set_index('reference_key')
                           .sort_index()
                           .to_xarray())
    
    ds_sumup = df_new.drop(columns=['name', 'method','reference','reference_short']).to_xarray()
    ds_meta = xr.merge((
                        ds_meta_name,
                        ds_meta_method,
                        ds_reference_method))
    
    ds_sumup['elevation'] = ds_sumup.elevation.astype(int)
    ds_sumup['error'] = ds_sumup['error'].astype(float)  
    ds_sumup['notes'] = ds_sumup['notes'].astype(str)      
    ds_meta['name'] = ds_meta['name'].astype(str)      
    ds_meta['method'] = ds_meta['method'].astype(str)      
    ds_meta['reference'] = ds_meta['reference'].astype(str)      
    ds_meta['reference_short'] = ds_meta['reference_short'].astype(str)      
    
    ds_sumup.timestamp.encoding['units'] = 'days since 1900-01-01'
    
    # attributes
    df_attr = pd.read_csv('doc/attributes_temperature.csv',
                          skipinitialspace=True,
                          comment='#').set_index('var')
    for v in df_attr.index:
        for c in df_attr.columns:
            if v in ds_sumup.keys():
                ds_sumup[v].attrs[c] = df_attr.loc[v,c]
            if v in ds_meta.keys():
                ds_meta[v].attrs[c] = df_attr.loc[v,c]
    if ds_sumup.latitude.isel(measurement_id=0)>0:
        ds_sumup.attrs['title'] = 'SUMup temperature dataset for the Greenland ice sheet (2023 release)'
    else:
        ds_sumup.attrs['title'] = 'SUMup temperature dataset for the Antarctica ice sheet (2023 release)'
    ds_sumup.attrs['contact'] = 'Baptiste Vandecrux'
    ds_sumup.attrs['email'] = 'bav@geus.dk'
    ds_sumup.attrs['production date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
       
    float_encoding = {"dtype": "float32", "zlib": True,"complevel": 9}
    int_encoding = {"dtype": "int32", "_FillValue":-9999, "zlib": True,"complevel": 9}
    
    ds_sumup[['name_key', 'reference_key', 'method_key', 'timestamp',
              'latitude', 'longitude', 'elevation',  
              'temperature',  'depth', 'duration','open_time',
              'error']].to_netcdf(filename, 
                                  group='DATA',
                                  encoding={
                                     "temperature": float_encoding |{'least_significant_digit':1},
                                     "depth": float_encoding |{'least_significant_digit':1},
                                     "timestamp": int_encoding,
                                     "duration": int_encoding,
                                     "open_time": int_encoding,
                                     "error": float_encoding|{'least_significant_digit':1},
                                     "longitude": float_encoding|{'least_significant_digit':6},
                                     "latitude": float_encoding|{'least_significant_digit':6},
                                     "elevation": int_encoding,
                                     "name_key": int_encoding,
                                     "reference_key": int_encoding,
                                     "method_key": int_encoding,
                                     })
    ds_meta.to_netcdf(filename, group='METADATA', mode='a',
                       encoding={
                               "name": {"zlib": True,"complevel": 9},
                               "reference": {"zlib": True,"complevel": 9},
                               "reference_short": {"zlib": True,"complevel": 9},
                               "method": {"zlib": True,"complevel": 9},
                           }
                      )
    
write_netcdf(df_sumup.loc[df_sumup.latitude>0, :], 'SUMup 2023 beta/SUMup_2023_temperature_greenland.nc')
write_netcdf(df_sumup.loc[df_sumup.latitude<0, :], 'SUMup 2023 beta/SUMup_2023_temperature_antarctica.nc')
#%% producing files for ReadMe file
import matplotlib.pyplot as plt
    
df_meta = pd.DataFrame()
df_meta['total'] = [df_sumup.shape[0]]
df_meta['added'] = df_sumup.shape[0]-len_sumup_2022
df_meta['nr_references'] = str(len(df_sumup.reference.unique()))
df_meta['greenland'] = df_sumup.loc[df_sumup.latitude>0].shape[0]
df_meta['antarctica'] = df_sumup.loc[df_sumup.latitude<0].shape[0]
df_meta.index.name='index'

df_meta.to_csv('doc/ReadMe_2023_src/tables/temperature_meta.csv')
print('  ')
print('{:,.0f}'.format(df_sumup.shape[0]).replace(',',' ') +\
      ' temperature observations in SUMup 2023')
print('{:,.0f}'.format(df_sumup.shape[0]-len_sumup_2022).replace(',',' ') +\
      ' more than in SUMup 2022')
print('from '+ str(len(df_sumup.reference_short.unique())) + ' sources')
print('representing '+ str(len(df_sumup.reference.unique()))+' references')

print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude>0].shape[0]).replace(',',' ')+' observations in Greenland')
print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude<0].shape[0]).replace(',',' ')+' observations in Antarctica')


plot_dataset_composition(df_sumup.loc[df_sumup.latitude>0], 'doc/ReadMe_2023_src/figures/temperature_dataset_composition_greenland.png')

plot_map(df_sumup.loc[df_sumup.latitude>0,['latitude','longitude']].drop_duplicates(),
         'doc/ReadMe_2023_src/figures/temperature_map_greenland.png', 
         area='greenland')

plot_dataset_composition(df_sumup.loc[df_sumup.latitude<0], 'doc/ReadMe_2023_src/figures/temperature_dataset_composition_antarctica.png')

plot_map(df_sumup.loc[df_sumup.latitude<0,['latitude','longitude']].drop_duplicates(),
         'doc/ReadMe_2023_src/figures/temperature_map_antarctica.png', 
         area='antarctica')


print_table_dataset_composition(df_sumup.loc[df_sumup.latitude>0]).to_csv('doc/ReadMe_2023_src/tables/composition_temperature_greenland.csv',index=None)

print_table_dataset_composition(df_sumup.loc[df_sumup.latitude<0]).to_csv('doc/ReadMe_2023_src/tables/composition_temperature_antarctica.csv',index=None)

print('writing out measurement locations')
print_location_file(df_sumup.loc[df_sumup.latitude>0,:], 
                    'doc/GIS/SUMup_2023_temperature_location_greenland.csv')

print_location_file(df_sumup.loc[df_sumup.latitude<0, :],
                    'doc/GIS/SUMup_2023_temperature_location_antarctica.csv')

