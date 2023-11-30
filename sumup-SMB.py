# -*- coding: utf-8 -*-
"""
SUMup compilation script

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
import re


def check_duplicates (df_stack, df_sumup, verbose=True, plot=True, tol=0.3):
    ''''
    Finding potential duplicates in new SMB data compared to what's in SUMup
    Input:
        df_stack:   dataframe with potentially multiple new SMB measurements to be
                    added to SUMup
        df_sumup:   dataframe with SUMup observations
        verbose:    if True print the candidates-duplicates
        plot:       if True print the candidates-duplicates
        tol:        tolerence, in degrees, accepted for coordinates of potential
                    duplicates compared to the location of new data
    Output:
        A dataframe containing the potential duplicates' name_str, reference_short, 
        latitude, longitude and elevation
    '''
    df_all_candidates = pd.DataFrame()
    for p in df_stack.name.unique():
        df = df_stack.loc[df_stack.name==p,:]
        
        # looking for obs within defined tolerence
        msk1 = abs(df_sumup.latitude - df.latitude.iloc[0])<tol
        msk2 = abs(df_sumup.longitude - df.longitude.iloc[0])<tol
        # looking for cores or snow pits and not radar profiles
        # this makes the following much faster
        msk3 = ~df_sumup.method.isin([2,6,7])
        msk = msk1 & msk2 & msk3
        if msk.any():
            # looping through the different observation groups (as defined by name_str)
            for p_dup in df_sumup.loc[msk, 'name_str'].unique():
                # extracting potential duplicate from SUMup
                df_sumup_candidate_dupl = df_sumup.loc[df_sumup.name_str == p_dup, :]
                # making a last check that the new data (in df) and the candidate
                # duplicate ends more or less at the same year
                # could make it more strict but I have seen that different sources 
                # variably include the very last year, and some time the record
                # is cropped if it comes from a study looking only at recent accumulation
                if abs(df_sumup_candidate_dupl.end_year.max() - df.end_year.max()) > 3:
                    # print(p_dup,df.name_str.iloc[0],'different end year')
                    continue
                
                # we append that candidate to the list of candidates
                df_all_candidates = pd.concat((
                    df_all_candidates,
                    df_sumup_candidate_dupl[
                        ['name_str','reference_short','latitude','longitude','elevation']
                        ].drop_duplicates()))
                
            if len(df_all_candidates)>=1:
                if verbose:
                    print('')
                    print(df_all_candidates.values)
                    print('might be the same as')
                    print(df[['name_str','latitude','longitude']].drop_duplicates().values)
                    if df.name_str.iloc[0] == 'sdo2':
                        print('sdo2 has same coordiantes as S.Domea and b but',
                              'different time range')
                if plot:
                    plt.figure()
                    plt.plot(df.end_year.values, df.smb.values, 
                              marker='o',ls='None', 
                              label=df.reference_short.iloc[0])
                    for p_dup in df_all_candidates.name_str.unique():
                        plt.plot(df_sumup.loc[df_sumup.name_str == p_dup, 'end_year'].values,
                                 df_sumup.loc[df_sumup.name_str == p_dup, 'smb'].values, 
                              marker='^',ls='None',alpha=0.7, 
                              label=(df_sumup.loc[df_sumup.name_str == p_dup, 
                                                 'name_str'].iloc[0]
                                     + ' ' +
                                     df_sumup.loc[df_sumup.name_str == p_dup, 
                                                        'reference_short'].iloc[0]))
                    plt.title(df.name_str.iloc[0])
                    plt.legend()
    return df_all_candidates

                
  
def resolve_reference_keys(df, df_sumup):
    '''
    Compares a dataframes 'reference_full' field  to the references that are already in
    SUMup. If a reference is already in SUMup (df_sumup), then it's reference_key
    is reused. If it is a new reference, then a reference_key is created.
    
    '''
    df_ref = df_sumup[['reference','reference_full']].drop_duplicates().set_index('reference')
    df_ref_to_add = (df
                        .reference_full.drop_duplicates()
                        .reset_index(drop=True)
                        .reset_index(drop=False)
                        .set_index('reference_full')
                        )
    count=1
    for r in df_ref_to_add.index:
        tmp = df_ref.reference_full.str.contains(r, regex=False)
        if tmp.any():
            # reference already in df_sumup, reusing key
            df_ref_to_add.loc[r, 'index'] = tmp.index[tmp.values].astype(int)[0]
        else:
            # new reference, creating new key
            df_ref_to_add.loc[r, 'index'] = df_sumup.reference.max() + count
            count = count+1
    return df_ref_to_add.loc[df.reference_full].values


def resolve_name_keys(df, df_sumup):
    '''
    Creates name_keys for all the unique 'name_str' in df.
    '''
    df_names_to_add = (df
                        .name_str.drop_duplicates()
                        .reset_index(drop=True)
                        .reset_index(drop=False)
                        .set_index('name_str')
                        )
    return df_names_to_add.loc[df.name_str].values + 1 + df_sumup.name.max()


# loading 2022 data
try:
    df_sumup = pd.read_csv('data/SUMup 2022/SUMup_accumulation_2022.csv')
except:
    # 2022 data too heavy for GitHub, it needs to be downloaded and stored locally
    print('Downloading SUMup 2022 accumulation file')
    url = 'https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A46c10a02-7184-47a2-ae9f-26e576010ffd'
    r = requests.get(url, allow_redirects=True)
    open('data/SUMup 2022/SUMup_accumulation_2022.csv', 'wb').write(r.content)    
    df_sumup = pd.read_csv('data/SUMup 2022/SUMup_accumulation_2022.csv')

df_sumup.columns = df_sumup.columns.str.lower()
df_sumup = df_sumup.rename(columns={'accumulation':'smb',
                                    'citation': 'reference',
                                    'radar_horizontal_resolution': 'notes'})

df_sumup[df_sumup==-9999] = np.nan

df_ref = pd.read_csv('data/SUMup 2022/SUMup_accumulation_references_2022.txt', sep='eirj', engine='python', header=None)
df_ref.columns = ['reference']
df_ref['key'] = np.arange(1,len(df_ref)+1)
df_ref = df_ref.set_index('key')

df_names = pd.read_csv('data/SUMup 2022/SUMup_accumulation_names_2022.txt', sep='eirj', engine='python', header=None)
df_names.columns = ['name']
df_names['key'] = np.arange(1,len(df_names)+1)
df_names = df_names.set_index('key')

df_methods = pd.read_csv('data/SUMup 2022/SUMup_accumulation_methods_2022.txt', header=None, sep='\t')
df_methods.columns = ['method']
df_methods.index = df_methods.index+1
df_methods.index.name = 'key'

df_sumup['name_str'] = np.nan
df_sumup.loc[df_sumup.name.notnull(), 
             'name_str'] = df_names.loc[df_sumup.loc[df_sumup.name.notnull(),
                              'name'] , 'name'].values
df_sumup['reference_full'] = df_ref.loc[df_sumup.reference, 'reference'].values
df_sumup['method_str'] = df_methods.loc[df_sumup.method, 'method'].values

# first guess of short reference, could be wrong
short_ref = parse_short_reference(df_sumup).set_index('reference')


for ref, s in zip(short_ref.index, short_ref.reference_short):

    if s.startswith('Ansch'): 
            short_ref.loc[ref, 'reference_short'] = 'Anschütz et al. (2007a,b,c,d,e,f)'  

    if s.startswith('Bolzan et al. (1999'): 
        short_ref.loc[ref, 'reference_short'] = \
            'Bolzan and Strobel (1999a,b,c,d,e,f,g,h,i,j,k,l,m,n,o)'   
    if s.startswith('Bolzan et al. (2001'):
        short_ref.loc[ref, 'reference_short'] = 'Bolzan and Strobel (2001a,b)'  
    if s.startswith('Graf et al. (2006'): 
        short_ref.loc[ref, 'reference_short'] = \
            'Graf and Oerter (2006a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z)'   
    if s.startswith('Graf et al. (1988'): 
        short_ref.loc[ref, 'reference_short'] = \
            'Graf et al. (1988a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q)'
    if s.startswith('Graf et al. (1999'): 
        short_ref.loc[ref, 'reference_short'] = \
            'Graf et al. (1999a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)'

    if s.startswith('Oerter et al. (2008'): 
        short_ref.loc[ref, 'reference_short'] = \
            'Oerter et al. (2008a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)'

    if s.startswith('Fernandoy'): 
        short_ref.loc[ref, 'reference_short'] = 'Fernandoy et al. (2010a,b,c,d)'
    if s.startswith('Wagenbach et al. (1994'): 
        short_ref.loc[ref, 'reference_short'] = 'Wagenbach et al. (1994a,b)'
        
    
df_sumup['reference_short'] = short_ref.loc[df_sumup.reference_full].reference_short.values
len_sumup_2022 = df_sumup.shape[0]

print(df_sumup.shape[0], 'SMB observations currently in SUMup from', len(df_sumup.reference_full.unique()), 'sources')
print(df_sumup.loc[df_sumup.latitude>0].shape[0], 'in Greenland')
print(df_sumup.loc[df_sumup.latitude<0].shape[0], 'in Antarctica')

# switching radar resolution as a note
df_sumup.loc[df_sumup.notes.notnull(), 'notes'] = 'Radar horizontal resolution: ' + df_sumup.loc[df_sumup.notes.notnull(), 'notes'].astype(str)

# adding field start_date, end_date for measurements that have that info
df_sumup['start_date'] = np.nan
df_sumup['end_date'] = np.nan

# If a measurement relates to a single year Y, then start_year = end_year = Y
df_sumup.loc[df_sumup.start_year.isnull(), 'start_year'] = \
    df_sumup.loc[df_sumup.start_year.isnull(), 'year'] 
df_sumup.loc[df_sumup.end_year.isnull(), 'end_year'] = \
    df_sumup.loc[df_sumup.end_year.isnull(), 'year'] 
# so we can drop column "year"
df_sumup = df_sumup.drop(columns='year')

df_sumup = df_sumup.drop(columns='timestamp')

col_needed = ['name', 'reference', 'method', 'start_date',
       'end_date', 'start_year','end_year', 'latitude', 'longitude', 'elevation',
       'notes', 'smb', 'error', 'name_str',
       'reference_full', 'method_str', 'reference_short']

# building names for pooled measurements at name = 25
# df_sumup.loc[df_sumup.name==25, ['latitude','longitude','reference_short']].drop_duplicates()

len_start = df_sumup.shape[0]

# Fixing error in the 2022 file
# in the PARCA cores
df_sumup.loc[26126, ['start_year','end_year']] = 1989 # instead of 1889
df_sumup.loc[26127, ['start_year','end_year']] = 1988 # instead of 1888

# PARCA-1997-cores (NASA East Core A) and PARCA-1997-cores (NASA East Core B)
# were put under the same name PARCA-1997-cores (NASA East Core A)
df_sumup.loc[26375:26404, 'name_str'] = 'PARCA-1997-cores (NASA East Core B)'
df_sumup.loc[26375:26404, 'name'] = df_sumup.name.max()+1

# some longitude in degW instead of negative degE in Greenland
df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'longitude'] = -df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'longitude']

# updating method for the PARCA cores
msk = df_sumup.name_str.astype(str).str.startswith('PARCA-1997')
df_sumup.loc[msk, 'method_str'] = 'firn or ice core, H2O2 dating'
df_sumup.loc[msk, 'method'] = 10
msk = df_sumup.name_str.astype(str).str.startswith('PARCA-1998')
df_sumup.loc[msk, 'method_str'] = 'firn or ice core, dO18 dating'
df_sumup.loc[msk, 'method'] = 11

# ACT10 cores have wrong years
df_sumup.loc[25579:25603,'start_year'] = np.arange(2009,1984,-1)
df_sumup.loc[25579:25603,'end_year'] = np.arange(2009,1984,-1)
df_sumup.loc[25604:25633, 'start_year'] = np.arange(2009,1979,-1)
df_sumup.loc[25604:25633, 'end_year'] = np.arange(2009,1979,-1)
df_sumup.loc[25634:25670, 'start_year'] = np.arange(2009,1972,-1)
df_sumup.loc[25634:25670, 'end_year'] = np.arange(2009,1972,-1)

df_sumup.loc[25579:25670, 'method'] = 5
df_sumup.loc[25579:25670, 'method_str'] = 'firn or ice core'


# %% Machguth SMB compilation
print('loading Machguth SMB compilation')
xl = pd.ExcelFile('data/SMB data/Machguth_SMB_database/greenland_SMB_database_v20200924.xlsx')
sheet_list = xl.sheet_names  # see all sheet names
meta = xl.parse('overview')  # read a specific sheet to DataFrame
meta = meta.iloc[:-1,:]

ref = xl.parse('references')  # read a specific sheet to DataFrame
ref.columns = ['code','ref','ref2','ref3']
ref.loc[ref.ref2.isnull(), 'ref2'] = ''
ref.loc[ref.ref3.isnull(), 'ref3'] = ''
ref['ref'] = ref.ref + ref.ref2 + ref.ref3
ref = ref.drop(columns=['ref2','ref3']).set_index('code').ref
ref = pd.concat((ref,pd.DataFrame(data=['machguth:16'], index=['machguth:16'])))

df_all = pd.DataFrame()

t_src_flag = {1:	'date known',  # exact date published, no need for flag
              2:	'only month published, middle of month chosen',
              3:	'no dates published, only given as winter or summer values',
              4:	'value refers to a fixed date system (exact start and '+\
                    'ending date known) and have been interpolated to the '+\
                    'fixed date system by modelling or by means of comparison '+\
                    'to other surface mass balance data',
              5:	'end date known, uncertain start date',
              6:	'date reconstructed from various sources'}
    
smb_source_type = {1: 'stake measurements' ,  # raw values in publication, no need for flag
              2: 'stake measurements, unclear whether snow or ice',
              3: 'estimated from mass balance profile',
              4: 'firn or ice core',
              5: 'pressure transducer'}

coordinates_source	= {1:	'coordinates known', # coordinates from measurements, no need for flag
                2:	'coordinates manually determined from map',
                3:	'coordinates determined from other information (e.g. distance from glacier tongue)'}

for sheet in sheet_list[4:]:
#     break
# #%%
    df = xl.parse(sheet)
    df = df.drop(columns =['X_o',	'Y_o',	'Z_o']).rename(
        columns = {'X':'longitude',
                              'Y':'latitude',
                              'Z':'elevation',
                              'H':'elevation',
                              'start':'start_date',
                              'end':'end_date',
                              'b':'smb',
                              'XY_src_flag': 'coordinates_source',
                              'XY_unc_flag': 'coordinates_uncertainty_m',
                              't_src_flag': 'date_source',
                              't_unc_flag': 'date_uncertainty_day',
                              'b_src_flag': 'smb_source_type',
                              'b_rho_flag': 'surface_density_assumption',
                              'b_unc_flag': 'error',  #'smb_uncertainty_dm_we',
                              'source_b': 'reference_full',
                              'point_ID': 'point_name',
                              })
    
    glacier_name = meta.set_index('glacier_ID').loc[int(sheet)].glacier_name
    df['glacier_name'] = glacier_name
    df['coordinates_source'] = [coordinates_source[v] for v in df.coordinates_source.values]
    df['date_source'] = [t_src_flag[v] for v in df.date_source.values]
    
    # method fields
    df['method_str'] = [smb_source_type[v] for v in df.smb_source_type.values]
    df_method_to_add = (df
                        .method_str.drop_duplicates()
                        .reset_index(drop=True)
                        .reset_index(drop=False)
                        .set_index('method_str')
                        )
    count=1
    for r in df_method_to_add.index:
        tmp = df_methods.method.str.contains(r)
        if tmp.any():
            df_method_to_add.loc[r, 'index'] = tmp.index[tmp.values].values[0]
        else:
            df_method_to_add.loc[r, 'index'] = df_sumup.method.max() + count
            count = count+1
    df['method'] = df_method_to_add.loc[df.method_str].values

    # creating observation name and key
    df['name_str'] = df.point_name.str.replace('_',' ').str.replace(sheet, glacier_name)
    df_names_to_add = (df
                        .name_str.drop_duplicates()
                        .reset_index(drop=True)
                        .reset_index(drop=False)
                        .set_index('name_str')
                        )
    df['name'] = df_names_to_add.loc[df.name_str].values + 1 + df_sumup.name.max()
    
    # creating reference and key

    # one ref missing in excel file
    if df.reference_full.isnull().sum()>0:
        if df.glacier_ID.iloc[0]==320:
            df.loc[df.reference_full.isnull(), 'reference_full'] = 'clement:81c'
        elif df.glacier_ID.iloc[0]==130:
            df.loc[df.reference_full.isnull(), 'reference_full'] = 'clausen-:01'
        else:
            
            print(df.loc[df.reference_full.isnull(), ['name_str','reference_full']] )
            print(wtf)
            
    # fixíng typo in excel file
    df['reference_full'] = df.reference_full.str.replace('estimate_based_on_','')
    df['reference_full'] = df.reference_full.str.replace('ACFEL:63','acfel:63')
    df['reference_full'] = df.reference_full.str.replace('ACFEL1963','acfel:63')
    df['reference_full'] = df.reference_full.str.replace('clement_83b','clement:83b')
    df['reference_full'] = df.reference_full.str.replace('clement:82a','clement-:82a')
    df['reference_full'] = df.reference_full.str.replace('data: Asiaq','data:asiaq')
    df['reference_full'] = df.reference_full.str.replace('braithwaite:82','braithwaite-:82')
    df['reference_full'] = df.reference_full.str.replace('drygalski-:1897','drygalski:1897a')
    df['reference_full'] = df.reference_full.str.replace(',densi',' densi')
    df['reference_full'] = df.reference_full.str.replace('own interpretation GC-Net data received 1/7/2014','steffen:online')
    tmp = df.reference_full.str.split(':')
    df.reference_full = tmp.str[0].str.split(' ').str[-1].str.lower() + ':' + tmp.str[1].str.split(' ').str[0]
    df['reference_full'] = df.reference_full.str.replace('nobles-:60','nobles:60')
    df['reference_full'] = df.reference_full.str.replace(',','')
    
    # switching from Horst's short reference system to full reference
    df['reference_full'] = ref.loc[df.reference_full.values].values[:,0]
    df['reference'] = resolve_reference_keys(df,  df_sumup)
    
    # rebuilding short reference
    df['reference_short'] = df['reference_full'].str.split(',').str[0] \
        + ', ' + df['reference_full'].str.split(',').str[-1].str.replace('.','',regex=False)
    
    # overwritting some outdated references
    df['reference_short'] = df.reference_short.str.replace(
        'Fausto,  Dataset published via Geological Survey of Denmark and '
        + 'Greenland DOI: https://doiorg/1022008/promice/data/aws',
        'PROMICE (Fausto et al., 2023)', regex=False)    
    
    df['reference_short'] = df.reference_short.str.replace(
        'http://cires.colorado.edu/science/groups/steffen/gcnet/ , '
        + 'http://cirescoloradoedu/science/groups/steffen/gcnet/ ',
        'GC-Net (Steffen et al., 2023)', regex=False)

    msk = (df.reference_full.str.lower().str.contains('unpubl',regex=False) | df.reference_full.str.lower().str.lower().str.contains('unknown'))
    df.loc[msk, 'reference_short'] = 'Machguth et al. (2016)'
    df.loc[msk, 'reference_full'] = 'Machguth, H., Thomsen, H.H., Weidick, A., Ahlstrøm, A.P., Abermann, J., Andersen, M.L., Andersen, S.B., Bjørk, A.A., Box, J.E., Braithwaite, R.J. and Bøggild, C.E., 2016. Greenland surface mass-balance observations from the ice-sheet ablation area and local glaciers. Journal of Glaciology, 62(235), pp.861-887. https://doi.org/10.1017/jog.2016.75'
    
    msk = ~msk
    df.loc[msk, 'reference_short'] = df.loc[msk, 'reference_short'] + ' as in Machguth et al. (2016)'
    df.loc[msk, 'reference_full'] = df.loc[msk, 'reference_full'] + ' as in Machguth, H., Thomsen, H.H., Weidick, A., Ahlstrøm, A.P., Abermann, J., Andersen, M.L., Andersen, S.B., Bjørk, A.A., Box, J.E., Braithwaite, R.J. and Bøggild, C.E., 2016. Greenland surface mass-balance observations from the ice-sheet ablation area and local glaciers. Journal of Glaciology, 62(235), pp.861-887. https://doi.org/10.1017/jog.2016.75'
    
    df.loc[df.reference_short.str.startswith('Clement,  1981'), 'reference_short'] = \
        'Clement (1981a,b,c,d) as in Machguth et al. (2016)'
    df.loc[df.reference_short.str.startswith('Clement,  1982'), 'reference_short'] = \
        'Clement (1982a,b,c) as in Machguth et al. (2016)'
    df.loc[df.reference_short.str.startswith('Clement,  1983'), 'reference_short'] = \
        'Clement (1983a,b,c,d) as in Machguth et al. (2016)'
    
    # creating note field with added info
    df['notes'] = (
                    df.coordinates_source + ' +/- '
                    + df.coordinates_uncertainty_m.astype(str) + ' m accuracy, '
                    + df.date_source + ' +/- '
                    + df.date_uncertainty_day.astype(str) + ' day, '
                    + 'density assumuption: '+ df.surface_density_assumption.astype(str) + ', '
                    + 'coordinates from'+ df.source_xy.astype(str).str.replace('nan','') + ', '
                    + 'elevation from'+ df.source_z.astype(str).str.replace('nan','') + ', '
                    + 'date from'+ df.source_t.astype(str).str.replace('nan','') + ', '
                    + df.remarks.astype(str).str.replace('nan','')
                    )
    
    df[['start_year', 'end_year']] = np.nan
    df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)


# %% Box compilation
print('Box compilation')
df = pd.read_excel('data/SMB data/Box compilation/Greenland_snow_pit_SWE_v20230104e_20230806.xlsx')

df = df.rename(columns={'lat':'latitude',
                        'lon':'longitude',
                        'Name':'name_str',
                        'Start Year':'start_year',
                        'End Year':'end_year',
                        'Elevation':'elevation',
                        'Accumulation':'smb',
                        'Accumulation error':'error',
                        'Type': 'method_str',
                        'Source':'reference_short',
                        'Publication': 'reference_full',
                        'Notes':'notes'                        
                        })
df['start_date'] = pd.to_datetime(df.TimeStart, errors='coerce')
df['end_date'] = pd.to_datetime(df.TimeEnd, errors='coerce')
df['longitude'] = -df.longitude.abs()
df['smb'] = df['smb']/1000
df['error'] = df['error']/1000

msk = df.name_str.str.lower().str.replace(' ','').isin(
    pd.DataFrame(['ACT-10a', 'ACT-10b', 'ACT-10c', 'ACT-11b', 'ACT-11c', 
                  'ACT-11d', 'Basin1', 'Basin2', 'Basin4', 'Basin5', 'Basin6',
                  'Basin7', 'Basin8', 'Basin9', 'CC2', 'D4', 'D5', 'Das1', 'Das2', 
                  'DYE-3_20D', 'GRIP89S1', 'McBales', 'Raven-Dust', 'Sandy', 
                  'Site15', 'Site6', 'Summit-Zoe-10',
                  'CC', 'CC2', 'NASA-E A', 'NASA-E B', 'NASA-U A',
                  'South Dome A', 'South Dome B','DYE-3_20D','GITS-1',
                  'GITS-2']
                  )[0].str.lower().str.replace(' ','')
    )
print('Removing', df.loc[msk,['name_str','reference_short']].drop_duplicates())
df = df.loc[~msk,:]

df = df.loc[df.smb.notnull()]

df.loc[df.reference_short.astype(str).str.startswith('Hermann'), 'reference_short'] =  \
    'Hermann et al. (2018)'
df.loc[df.reference_short.astype(str).str.startswith('Hermann'), 'reference_full'] =  \
 'Hermann, M., Box, J. E., Fausto, R. S., Colgan, W. T., Langen, P. L., Mottram, R., et al. (2018). Application of PROMICE Q-transect in situ accumulation and ablation measurements (2000–2017) to constrain mass balance at the southern tip of the Greenland ice sheet. Journal of Geophysical Research: Earth Surface, 123, 1235–1256. https://doi.org/10.1029/2017JF004408'

df.loc[df.reference_short.astype(str).str.startswith('Kjær'), 'reference_short'] =  \
    'Kjær et al. (2021)'
df.loc[df.reference_short.astype(str).str.startswith('Kjær'), 'reference_full'] =  \
'Kjær, H. A., Zens, P., Edwards, R., Olesen, M., Mottram, R., Lewis, G., Terkelsen Holme, C., Black, S., Holst Lund, K., Schmidt, M., Dahl-Jensen, D., Vinther, B., Svensson, A., Karlsson, N., Box, J. E., Kipfstuhl, S., and Vallelonga, P.: Recent North Greenland temperature warming and accumulation, The Cryosphere Discuss. [preprint], https://doi.org/10.5194/tc-2020-337, 2021.'


df.loc[df.name_str.astype(str).str.startswith('NANOK'), 'reference_full'] =  'GEUS unpublished'
df.loc[df.name_str.astype(str).str.startswith('NANOK'), 'reference_short'] =  'GEUS unpublished'
       
df.loc[df.reference_short == 'Box/Wei/Mosley-Thompson', 'reference_short'] =  \
    'Burgress et al. (2010)'
df.loc[df.reference_full == 'Box/Wei/Mosley-Thompson', 'reference_full'] =  \
'Burgess, E. W., Forster, R. R., Box, J. E., Mosley-Thompson, E., Bromwich, D. H., Bales, R. C., and Smith, L. C. (2010), A spatially calibrated model of annual accumulation rate on the Greenland Ice Sheet (1958–2007), J. Geophys. Res., 115, F02004, doi:10.1029/2009JF001293. '

df.loc[df.reference_short == 'Schaller et al 2016', 'reference_short'] =  \
    'Schaller et al. (2016)'
df.loc[df.reference_short == 'Schaller et al. (2016)', 'reference_full'] =  \
'Schaller, C. F., Freitag, J., Kipfstuhl, S., Laepple, T., Steen-Larsen, H. C., and Eisen, O.: A representative density profile of the North Greenland snowpack, The Cryosphere, 10, 1991–2002, https://doi.org/10.5194/tc-10-1991-2016, 2016.'

df.loc[df.reference_short.astype(str).str.startswith('Niwano et al. (2020)'),
       'reference_short'] =  'Niwano et al. (2020)'


df.loc[df.notes.isnull(), 'notes'] =  ''

df.loc[df.reference_short.isnull(), 'reference_full'] =  'GEUS unpublished'
df.loc[df.reference_short.isnull(), 'reference_short'] =  'GEUS unpublished'

df.loc[df.reference_full.isnull(), 'reference_short'] =  'GEUS unpublished'
df.loc[df.reference_full.isnull(), 'reference_full'] =  'GEUS unpublished'
    
for ref in ['Box/GEUS Q-transect', 'Steffen/Box/Albert/Cullen/Huff/Weber/Starkweather/Molotch/Vandecrux',
            'Colgan/GEUS', 'Steffen/Cullen/Huff/Colgan/Box/Vandecrux', 'GEUS unpublished',
            'Box/Niwano', 'Braithwaite-GGU', 'ACT-PROMICE', 'Summit']:
    df.loc[df.reference_short == ref, 'notes'] =  df.loc[df.reference_short == ref, 'notes'].astype(str) + ' from ' + ref
    df.loc[df.reference_short == ref, 'reference_full'] = 'GEUS unpublished'
    df.loc[df.reference_short == ref, 'reference_short'] =  'GEUS unpublished'


df['reference'] = resolve_reference_keys(df, df_sumup)

df['name'] = resolve_name_keys(df, df_sumup)
df['method'] = -9999
df.loc[df.Regime=='ablation',  'method'] = 3
df.loc[df.Regime=='ablation',  'method_str'] = 'stake measurements'
df.loc[df.method_str=='core',  'method'] = 5
df.loc[df.method_str=='core',  'method_str'] = 'firn or ice core'
df.loc[df.method_str=='pit',  'method'] = 4
df.loc[df.method_str=='pit',  'method_str'] = 'snow pits'
# check_duplicates (df, df_sumup, plot=False)

df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)


# %% Lewis et al. 2017
print('Lewis et al. 2017')
df = pd.read_excel('data/SMB data/Lewis et al. 2017/ALL ICEBRIDGE DATA.xlsx')
df.columns = df.iloc[0,:].str.lower()
df=df.iloc[1:,:]

df_stack = df[df.columns[2:]].stack().to_frame()
df_stack.index = df_stack.index.rename(['id','date'])
df_stack.columns = ['smb']
df_stack = df_stack.reset_index()
df_stack['latitude'] = df.loc[df_stack.id.values,'latitude'].values
df_stack['longitude'] = df.loc[df_stack.id.values,'longtidue'].values
df_stack['end_year'] = df_stack.date.str.split('-').str[0].astype(int)
df_stack['start_year'] = df_stack.date.str.split('-').str[1].astype(int)
df_stack[['start_date', 'end_date']] = np.nan

df_stack['notes'] = 'value given for summer reflection to summer reflection (exact date unknown)'

df_stack['name_str'] = 'Lewis_2017_point_'+df_stack.id.astype(str)
df_stack['method'] = 6
df_stack['elevation'] = -9999
df_stack['error'] = np.nan
df_stack['method_str'] = 'airborne radar'
df_stack['reference_short'] = 'Lewis et al. (2017)'
df_stack['reference_full'] = 'Lewis, G., Osterberg, E., Hawley, R., Whitmore, B., Marshall, H. P., and Box, J.: Regional Greenland accumulation variability from Operation IceBridge airborne accumulation radar, The Cryosphere, 11, 773–788, https://doi.org/10.5194/tc-11-773-2017, 2017.'
df_stack['name'] = resolve_name_keys(df_stack, df_sumup)
df_stack['reference'] = resolve_reference_keys(df_stack, df_sumup)

df_sumup = pd.concat((df_sumup, df_stack[col_needed]), ignore_index=True)


# %% Montgomery et al. 2020
print('Montgomery et al. 2020')
df = pd.read_csv('data/SMB data/Montgomery et al. 2020/SEGL_accumulation_2009_2017.csv', low_memory=False)
df.columns = df.columns.str.lower()
df= df.rename(columns= {'accumulation (m w.e.)': 'smb'})
df.year = pd.to_numeric(df.year, errors='coerce')
df = df.loc[df.year.notnull(),:]
df['start_year'] = df.year.astype(int)
df['end_year'] = df.year.astype(int)
df[['start_date', 'end_date']] = np.nan

df['notes'] = 'value given from end of last summer (uncertain) to measurement time in spring (not available in data file)'

df['name_str'] = 'Montgomery_2020_point_'+df.index.astype(str)
df['method'] = 6
df['elevation'] = -9999
df['error'] = np.nan
df['method_str'] = 'airborne radar'
df['reference_short'] = 'Montgomery et al. (2020)'
df['reference_full'] = 'Montgomery L, Koenig L, Lenaerts JTM, Kuipers Munneke P (2020). Accumulation rates (2009–2017) in Southeast Greenland derived from airborne snow radar and comparison with regional climate models. Annals of Glaciology 61(81), 225–233. https://doi.org/10.1017/aog.2020.8, Data: Montgomery L, Koenig L, Lenaerts JTM, Kuipers Munneke P (2020). Southeast Greenland Accumulation Rates Derived from Operation IceBridge Snow Radar, 2009-2017. Arctic Data Center. doi:10.18739/A2J96095Z.'
df['name'] = resolve_name_keys(df, df_sumup)
df['reference'] = resolve_reference_keys(df, df_sumup)

df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# %% Lewis et al. 2019 GPR
print('Lewis et al. 2019')
df = pd.read_csv('data/SMB data/GreenTracs/Total_accumulation.csv', skiprows=2)
df.columns = df.columns.str.lower()

df_stack = df[df.columns[2:]].stack().to_frame()
df_stack.index = df_stack.index.rename(['id','date'])
df_stack.columns = ['smb']
df_stack = df_stack.reset_index()
df_stack['latitude'] = df.loc[df_stack.id.values,'latitude'].values
df_stack['longitude'] = df.loc[df_stack.id.values,'longitude'].values
df_stack['start_year'] = df_stack.date.str.split('-').str[0].astype(int)
df_stack['end_year'] = df_stack.date.str.split('-').str[1].astype(int)
df_stack[['start_date', 'end_date']] = np.nan

df_stack['notes'] = ''

df_stack['name_str'] = 'Lewis_2019_point_'+df_stack.id.astype(str)
df_stack['method'] = 7
df_stack['elevation'] = -9999
df_stack['error'] = np.nan
df_stack['method_str'] = 'ground-based radar'
df_stack['reference'] = 90
short_ref.loc[df_ref.loc[90], 'reference_short'] = 'Lewis et al. (2019)'
df_sumup.loc[df_sumup.reference == 90, 'reference_short'] = 'Lewis et al. (2019)'

df_stack['reference_short'] = short_ref.loc[df_ref.loc[90], 'reference_short'].values[0]
df_stack['reference_full'] = df_ref.loc[90].values[0]
df_stack['name'] = resolve_name_keys(df_stack, df_sumup)
df_stack['reference'] = resolve_reference_keys(df_stack, df_sumup)

df_sumup = pd.concat((df_sumup, df_stack[col_needed]), ignore_index=True)

# %% AWI NGT data
print('Freitag and Vinther NGT')

for f in os.listdir('data/SMB data/AWI NGT'):
    df_meta = pd.read_csv('data/SMB data/AWI NGT/'+f, sep='£',engine='python',header=None)
    skiprows = df_meta.index.values[df_meta[0] == '*/'][0]+1
    row = df_meta.index.values[df_meta[0].str.startswith('Event')][0]

    df = pd.read_csv('data/SMB data/AWI NGT/'+f, sep='\t', skiprows=skiprows)
    df.columns =  ['age', 'start_year','smb']

    df['reference_full'] = df_meta.iloc[1].values[0].split('\t')[1]
    df['latitude'] = float(re.findall("(?<=LATITUDE: )\d+\.\d+", df_meta.iloc[row].values[0])[0])
    df['longitude'] = -float(re.findall("(?<=LONGITUDE: -)\d+\.\d+", df_meta.iloc[row].values[0])[0])
    try:
        df['elevation'] = float(re.findall("(?<=ELEVATION: )\d+\.\d+", df_meta.iloc[row].values[0])[0])
    except Exception as e:
        print('Cannot find elevation')
        df['elevation'] = np.nan
    df['name_str'] = df_meta.iloc[row].values[0].split('\t')[1].split(' ')[0]
    
    df['end_year'] = df.start_year
    df[['start_date', 'end_date']] = np.nan
    df['notes'] = ''
    df['method'] = 5
    df['error'] = np.nan
    df['method_str'] = 'firn or ice core'
    if df_meta.iloc[1].values[0].split('\t')[1].startswith('Frei'):
        df['reference_short'] = 'Freitag et al. (2022a,b,c,d)'
    else:
        df['reference_short'] = 'Vinther et al. (2022)'

    df['name'] = resolve_name_keys(df, df_sumup)
    df['reference'] = resolve_reference_keys(df, df_sumup)
    print(df[['name_str', 'latitude','longitude','elevation', 'reference_short','name','reference']].drop_duplicates().values)
    
    df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)

# %% Miller and Schwager 2004 NGT
print('Miller and Schwager 2004 NGT')

for f in os.listdir('data/SMB data/Miller and Schwager 2004 NGT'):
    df_meta = pd.read_csv('data/SMB data/Miller and Schwager 2004 NGT/'+f, sep='£',engine='python',header=None)
    skiprows = df_meta.index.values[df_meta[0] == '*/'][0]+1
    row = df_meta.index.values[df_meta[0].str.startswith('Event')][0]

    df = pd.read_csv('data/SMB data/Miller and Schwager 2004 NGT/'+f, sep='\t', skiprows=skiprows)
    if f == 'ngt03c93_2_acc_d18O.tab':
        df.columns =  ['age', 'depth', 'start_year', 'dO18', 'depth_we', 'smb']
    else:
        df.columns =  ['depth', 'age', 'start_year', 'dO18', 'smb', 'depth_we']

    df['reference_full'] = df_meta.iloc[1].values[0].split('\t')[1]
    df['latitude'] = float(re.findall("(?<=LATITUDE: )\d+\.\d+", df_meta.iloc[row].values[0])[0])
    df['longitude'] = -float(re.findall("(?<=LONGITUDE: -)\d+\.\d+", df_meta.iloc[row].values[0])[0])
    try:
        df['elevation'] = float(re.findall("(?<=ELEVATION: )\d+\.\d+", df_meta.iloc[row].values[0])[0])
    except Exception as e:
        print('Cannot find elevation')
        df['elevation'] = np.nan
    df['name_str'] = df_meta.iloc[row].values[0].split('\t')[1].split(' ')[0]
    
    df['end_year'] = df.start_year
    df[['start_date', 'end_date']] = np.nan
    df['notes'] = ''
    df['smb'] = df['smb']/1000
    df['method'] = 5
    df['error'] = np.nan
    df['method_str'] = 'firn or ice core'
    df['reference_short'] = 'Miller and Schwager (2000a,b,c,d,e)'

    df['name'] = resolve_name_keys(df, df_sumup)
    df['reference'] = resolve_reference_keys(df, df_sumup)
    print(df[['name_str', 'latitude','longitude','elevation', 'reference_short','name','reference']].drop_duplicates().values)
    
    df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)
# %% Weissbach_2016
# Smoothed profiles: 
# print('Weissbach_2016 NGT')

# for f in os.listdir('data/SMB data/Weissbach_2016/datasets'):
#     if f.startswith('B27') | f.startswith('NG'):
#         continue
#     df_meta = pd.read_csv('data/SMB data/Weissbach_2016/datasets/'+f, sep='£',engine='python',header=None)
#     skiprows = df_meta.index.values[df_meta[0] == '*/'][0]+1
#     row = df_meta.index.values[df_meta[0].str.startswith('Event')][0]

#     df = pd.read_csv('data/SMB data/Weissbach_2016/datasets/'+f, sep='\t', skiprows=skiprows)
#     df.columns =  ['age', 'start_year', 'depth1','depth_we', 'smb_smooth', 'do18']

#     df['reference_full'] = df_meta.iloc[1].values[0].split('\t')[1]
#     df['latitude'] = float(re.findall("(?<=LATITUDE: )\d+\.\d+", df_meta.iloc[row].values[0])[0])
#     df['longitude'] = -float(re.findall("(?<=LONGITUDE: -)\d+\.\d+", df_meta.iloc[row].values[0])[0])
#     try:
#         df['elevation'] = float(re.findall("(?<=ELEVATION: )\d+\.\d+", df_meta.iloc[row].values[0])[0])
#     except Exception as e:
#         print('Cannot find elevation')
#         df['elevation'] = np.nan
#     df['name_str'] = df_meta.iloc[row].values[0].split('\t')[1].split(' ')[0]
    
#     df['end_year'] = df.start_year
#     df[['start_date', 'end_date']] = np.nan
#     df['smb'] = np.nan
#     # df['smb'] = df['depth_we'].diff()
#     print(wtf)
#     df['notes'] = ''
#     df['method'] = 5
#     df['error'] = np.nan
#     df['method_str'] = 'firn or ice core'
#     df['reference_short'] = 'Weissbach et al. (2016a,b,c,d,e,f,g,h,i,j,k,l,m,n,o)'

#     df['name'] = resolve_name_keys(df, df_sumup)
#     df['reference'] = resolve_reference_keys(df, df_ref, df_sumup)
#     print(df[['name_str', 'latitude','longitude','elevation', 'reference_short','name','reference']].drop_duplicates().values)
    
#     df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)
    
# %% Karlsson et al. 2016
print('Karlsson et al. 2016')
df_meta = pd.read_csv('data/SMB data/Karlsson et al. 2016/North_Greenland_acc_rates.tab', sep='£',engine='python',header=None)
skiprows = df_meta.index.values[df_meta[0] == '*/'][0]+1
row = df_meta.index.values[df_meta[0].str.startswith('Event')][0]

df = pd.read_csv('data/SMB data/Karlsson et al. 2016/North_Greenland_acc_rates.tab', sep='\t', skiprows=skiprows)
df.columns =  ['latitude', 'longitude', '1911-2011', '1811-1911', '1711-1811',
       '1611-1711', '1511-1611', '1411-1511', '1311-1411']


df_stack = df[df.columns[2:]].stack().to_frame()
df_stack.index = df_stack.index.rename(['id','date'])
df_stack.columns = ['smb']
df_stack = df_stack.reset_index()
df_stack['latitude'] = df.loc[df_stack.id.values,'latitude'].values
df_stack['longitude'] = df.loc[df_stack.id.values,'longitude'].values
df_stack['start_year'] = df_stack.date.str.split('-').str[0].astype(int)
df_stack['end_year'] = df_stack.date.str.split('-').str[1].astype(int)
df_stack[['start_date', 'end_date']] = np.nan

df_stack['notes'] = ''

df_stack['name_str'] = 'Karlsson_2016_point_'+df_stack.id.astype(str)
df_stack['smb'] = df_stack['smb']/1000
df_stack['method'] = 7
df_stack['elevation'] = np.nan
df_stack['error'] = np.nan
df_stack['method_str'] = 'ground-based radar'
df_stack['reference_short'] = 'Karlsson et al. (2016)'
df_stack['reference_full'] = df_meta.iloc[1].values[0].split('\t')[1]
df_stack['name'] = resolve_name_keys(df_stack, df_sumup)
df_stack['reference'] = resolve_reference_keys(df_stack, df_sumup)

df_sumup = pd.concat((df_sumup, df_stack[col_needed]), ignore_index=True)

# %% Box 2013 compilation
print('Box 2013 compilation')
ref_box = pd.read_excel('data/SMB data/Box 2013 compilation/ref.xlsx')

for f in os.listdir('data/SMB data/Box 2013 compilation/individual_formatted_accum_records/'):
    if not f.endswith('.txt'): continue
    if f.startswith('0_Sukkertoppen'): continue
    if f.startswith('NAO_'): continue
    if f.startswith('0temp'): continue
    df_meta=pd.read_csv(
        'data/SMB data/Box 2013 compilation/individual_formatted_accum_records/'+f,
                        sep='£',engine='python',header=None).iloc[:3,:]
    name = df_meta.iloc[0].values[0]
    if isinstance(name,float): name=str(int(name))
    name = name.replace(' ','')
    if name.startswith('D-'): name=name.replace('D-','D')

    if f == 'Tunu-N_accum_rate.txt': name = 'Tunu-N'
    if f == 'DYE-3-18C_accum_rate.txt': name = 'DYE-3-18C'
    if f == 'Raven-DO18_accum_rate.txt': name = 'Raven-DO18'
    if f == 'DYE-3-20D_accum_rate.txt': name = 'DYE-3-20D'
    if not f.replace(' ','').startswith(name.replace(' ','')): print(wtf)
    if name == 'N.DYE-2': name = 'N.DYE-3a'
    if name == 'N.DYE-3': name = 'N.DYE-3b'
    if name in ['ACT-10a','ACT-10b','ACT-10c']:
        print('Skipping',name,'already in SUMup 2022')
        continue
    lat = float(df_meta.iloc[1].str.extract(r'(\d+\.?\d*)').values[0][0])
    lon = float(df_meta.iloc[1].str.extract(r'(-\d+\.?\d*)')[0][0])
    
    # correcting coordinates using Bales et al. 2009 and Hanna et al. 2006
    if name == 'Basin2': lat = 68.3; lon = -44.8
    if name == 'Basin6': lat = 67; lon = -41.7
    if name == 'Basin7': lat = 67.5; lon = -40.4

    df = pd.read_csv('data/SMB data/Box 2013 compilation/individual_formatted_accum_records/'
                     +f,skiprows=3,delim_whitespace=True)
    df.columns = ['start_year','smb']

    df['name_str'] = name
    df['latitude'] = lat
    df['longitude'] = lon
    df['elevation'] = np.nan
    df['end_year'] = df.start_year

    df['reference_short'] = ref_box.loc[
        ref_box.name.astype(str).str.lower().str.replace(' ','') == name.lower().replace(' ',''),
        'short_ref'].values[0]
    df['reference_full'] = ref_box.loc[
        ref_box.name.astype(str).str.lower().str.replace(' ','') == name.lower().replace(' ',''),
        'long_ref'].values[0]
                
    df[['start_date', 'end_date']] = np.nan
    df['notes'] = ''
    df['error'] = np.nan
    df['method'] = 5
    df['method_str'] = 'firn or ice core'
    if name.endswith('Dust'):
        df['method'] = 12
        df['method_str'] = 'firn or ice core, dust dating'
    if name.endswith('DO18'):
        df['method'] = 11
        df['method_str'] = 'firn or ice core, dO18 dating'

    df['name'] = resolve_name_keys(df, df_sumup)
    df['reference'] = resolve_reference_keys(df, df_sumup)
        
    if df.name_str.iloc[0] in ['GRIP93S1','GRIP','GRIP92S1','GRIP91S1','GRIP89S2',
                               'GRIP89S1','Milcent','SiteE','SiteG','SiteD',
                               'SiteB','SiteA','NGRIP','DYE-3','DYE-3-18C',
                               'DYE-3-20D','DYE-3_18C','Crete']:
        fig = plt.figure(figsize=(8,8))
        plt.plot(df.end_year,df.smb,marker='o',ls='None')
        plt.title(df.name_str.iloc[0]+' as in Box et al. (2013)')
        plt.ylabel('Accumulation (m w.e.)')
        plt.xlabel('Year')
        fig.savefig('figures/problematic/'+df.name_str.iloc[0]+'_in_Box_2013')
        plt.close(fig)
        print('Skipping',df.name_str.iloc[0],'due to quality issue')
        continue

    already_in = check_duplicates(df, df_sumup, verbose = False, plot=False, tol=0.6)
    if (name not in  ['GITS-2', 'S.Domeb']) & (not name.startswith('Hum')):
        if len(already_in)>=1:
            if already_in.reference_short.str.startswith('Mosley').any():
                # print('')
                # print('Replacing')
                # print(df[['name_str','latitude','longitude','elevation']]
                #           .drop_duplicates().to_markdown())
                ind_already_in = already_in.loc[
                    already_in.reference_short.str.startswith('Mosley'),:
                        ].index
                if (name[-1] == 'b') & (len(ind_already_in)>1): 
                    ind_already_in = [ind_already_in[1]]
                if (name[-1] == 'c') & (len(ind_already_in)>1):
                    ind_already_in = [ind_already_in[2]]
                df['name_str'] = df_sumup.loc[ind_already_in,'name_str'].values[0]+'_dust'
                df['latitude'] = df_sumup.loc[ind_already_in,'latitude'].values[0]
                df['longitude'] = df_sumup.loc[ind_already_in,'longitude'].values[0]
                df['elevation'] = df_sumup.loc[ind_already_in,'elevation'].values[0]
                df['reference'] = df_sumup.loc[ind_already_in,'reference'].values[0]
                df['reference_short'] = df_sumup.loc[ind_already_in,
                                                     'reference_short'].values[0]
                df['reference_full'] = df_sumup.loc[ind_already_in,
                                                    'reference_full'].values[0]
                df['method'] = 12
                df['method_str'] = 'firn or ice core, dust dating'
                # print('by')
                # print(df[
                #     ['name_str','latitude','longitude','elevation']
                #             ].drop_duplicates().to_markdown())
        #     else:
        #         check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
        # else:
        #     check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
    print(name, df.name_str.unique())
    df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)    

# %% Hanna 2006 compilation
print('Hanna 2006 compilation')

df = pd.read_excel('data/SMB data/Hanna 2006 compilation/accum.xls',
                      sheet_name='ann_accum_matrix_eh')
df = df.iloc[np.arange(0,116,2),:]
df = df.reset_index(drop=True)
df_meta = df.iloc[:,[0,2,3]]
df_meta.columns = ['name','latitude','longitude']
df = df.iloc[:,8:-7]
df = df.reset_index(drop=True)
df_stack = df.stack().reset_index()
df_stack.columns = ['id','start_year','smb']
df_stack['latitude'] = df_meta.loc[df_stack.id.values,'latitude'].values
df_stack['longitude'] = df_meta.loc[df_stack.id.values,'longitude'].values
df_stack['name_str'] = df_meta.loc[df_stack.id.values,'name'].values
df_stack['smb'] = df_stack.smb/100
df_stack['name'] = df_sumup.name.max()+df_stack.id.values+1
df_stack['reference_full'] = 'Hanna, E.P., J. McConnell, S. Das, J. Cappelen, and A. Stephens, 2006: Observed and modelled Greenland Ice Sheet snow accumulation, 1958–2003, and links with regional climate forcing. J. Climate, 19, 344–358.'
df_stack['reference_short'] = 'Hanna et al. (2006)'

df_stack['elevation'] = np.nan
df_stack['end_year'] = df_stack.start_year
df_stack[['start_date', 'end_date']] = np.nan
df_stack['notes'] = ''
df_stack['method'] = 5
df_stack['error'] = np.nan
df_stack['method_str'] = 'firn or ice core'
df_stack['reference'] = resolve_reference_keys(df_stack, df_sumup)

# the following profiles are exact copies of the accumulation values
# already in SUMup 2022 (PARCA H2O2 values) or Box 2013
msk = df_stack.name_str.astype(str).str.lower().str.replace(' ','').isin(
    ['basin9','basin8', 'basin1','das2','nasaea','saddlea', '7147',
     'sdomea', '7249', '6945', '6345', '7653', '7551', '7247', '7147',
     'stunua', 'basin2', 'basin6', 'basin7'])
df_stack = df_stack.loc[~msk,:]
list_remove =[]
for p in df_stack.name_str.unique():
    # # #%%
    # p = 7147
    
    df_all_candidates = check_duplicates (df_stack.loc[df_stack.name_str.astype(str)==str(p),:], 
                                          df_sumup, verbose=False, plot=True,
                                          tol=0.5)
    if len(df_all_candidates)==0:
        print('')
        print(p,'no duplicates detected')
    else:
        ind_candidate = (df_all_candidates.reference_short.str.startswith('Mosley') |
                df_all_candidates.reference_short.str.startswith('Bales')|
                df_all_candidates.reference_short.str.startswith('Banta')|
                df_all_candidates.reference_short.str.startswith('Hanna et al. (2006)'))
        if ind_candidate.any():
            ind_to_update = df_stack.index[df_stack.name_str.astype(str)==str(p)]
            print('')
            print('in Hanna 2006:')
            print(df_stack.loc[ind_to_update,
                               ['name_str', 'latitude','longitude','elevation']
                               ].drop_duplicates().to_markdown())
            for i, v in enumerate(['name_str','reference_short', 'latitude','longitude','elevation']):
                if v == 'reference_short': continue
                if v == 'name_str':
                    df_stack.loc[ind_to_update,v] = \
                        df_stack.loc[ind_to_update,v].astype(str) + '_in_Hanna_2006'
                else:
                    df_stack.loc[ind_to_update ,v] =\
                        df_all_candidates.loc[ind_candidate].iloc[0,i]
            print('')
            print('added to SUMup 2023 as:')
            print(df_stack.loc[ind_to_update,
                               ['name_str','latitude','longitude','elevation']
                               ].drop_duplicates().to_markdown())
        else:
            print('')
            print(df_stack.loc[df_stack.name_str.astype(str)==str(p),
                               ['name_str', 'latitude','longitude','elevation']
                               ].drop_duplicates().to_markdown())
            print('matching with')
            print(df_all_candidates.to_markdown())
            print('===> adding it anyway')
    
df_sumup = pd.concat((df_sumup, df_stack[col_needed]), ignore_index=True)    

# %% NSIDC files
# not added because of degraded resolution
plt.close('all')
for f in os.listdir('data/SMB data/NSIDC_Crete_Milcent_Summit_Osman'):
    if not f.endswith('.txt'): continue
    df_meta = pd.read_csv('data/SMB data/NSIDC_Crete_Milcent_Summit_Osman/'+f,
                          sep='£', engine='python', header=None,
                          skip_blank_lines=False)
    df_meta.loc[df_meta[0].isnull(), 0] = ''
    ind_start = df_meta[0].str.startswith('DATA:')[::-1].idxmax()+1
    if ind_start == len(df_meta[0]):
        ind_start = df_meta[0].str.startswith('#')[::-1].idxmax()+1
    df = pd.read_csv('data/SMB data/NSIDC_Crete_Milcent_Summit_Osman/'+f,
                     comment='#',
                      skiprows=ind_start+1, sep='\t', header=None)
    try:
        df.columns = ['end_year','smb']
    except:
        df.columns = ['end_year','smb','','','','','','','','']
    df=df.loc[df.end_year>1000,:]
    
    if f == 'act2-2021accum.txt':
        print('skipping', f)
        print('already in Box 2013')
    if 'accum.txt' not in f:
        print('skipping', f)
        print('due to degraded resolution')
        fig=plt.figure(figsize=(15,10))
        df.set_index('end_year').smb.plot(ax=plt.gca(), marker='o', ls='None')
        plt.title(f+' from NSIDC')
        plt.ylabel('Accumulation (cm ice yr-1)')
        plt.xlabel('year')
        fig.savefig('figures/problematic/'+f.replace('.txt','')+'_from_NSIDC')
        plt.close(fig)
        continue
    l = np.array([['# Site_Name:', 'name_str'],
            ['# Northernmost_Latitude:', 'latitude'],
            ['# Easternmost_Longitude:', 'longitude'],
            ['# Elevation:', 'elevation'],
            ])
    for pat, v in zip(l[:,0], l[:,1]):
        df[v] = df_meta.loc[df_meta[0].str.startswith(pat),
                                     0].iloc[0].split(': ')[1]
        if v != 'name_str':
            df[v] = df[v].astype(float)
    print(df[['name_str','latitude','longitude','elevation']].drop_duplicates())
    df['reference_full'] = 'Osman, M.B.; Coats, S.; Das, S.B.; McConnell, J.R.; Chellman, N. 2021. North Atlantic jet stream projections in the context of the past 1,250 years. Proceedings of the National Academy of Sciences, 118(38), e2104105118. doi: 10.1073/pnas.2104105118. Data: Osman, M.B.; Coats, S.; Das, S.B.; McConnell, J.R.; Chellman, N. (2021-09-17): NOAA/WDS Paleoclimatology - Greenland Ice Cores 1,250 Year d18O, Accumulation, and North Atlantic Jet Stream Reconstruction. [indicate subset used]. NOAA National Centers for Environmental Information. https://doi.org/10.25921/yx6v-ak24'
    
    df['reference_short'] = 'Osman et al. (2021)'
    df['error'] = np.nan
    df['smb'] = df.smb/1000
    df['start_year'] = df.end_year
    df[['start_date','end_date']] = np.nan
    df['notes'] = ''
    df['method_str'] = 'firn core'
    df['method'] = 5
    df['name'] = resolve_name_keys(df, df_sumup)
    df['reference'] = resolve_reference_keys(df, df_sumup)
    check_duplicates(df, df_sumup, verbose = True, plot=True, tol=0.6)
    df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)    


# %% SE Dome II 
print('Kawakami and Iizuka data from SE Dome')
df_meta = pd.read_csv('data/SMB data/Kawakami/kawakami2023-se-dome_accum_rate.txt',
                      sep='£', engine='python', header=None,
                      skip_blank_lines=False,encoding='mbcs')
df_meta.loc[df_meta[0].isnull(), 0] = ''
ind_start = df_meta.index[df_meta[0].str.startswith('# Missing_Values: ')].values[0]+1
df = pd.read_csv('data/SMB data/Kawakami/kawakami2023-se-dome_accum_rate.txt',
                  delim_whitespace=True,  skiprows=ind_start, encoding='mbcs')
df.columns = ['end_year','smb']

df['name_str'] = 'SE Dome II 2020'

df['latitude'] = 67.19
df['longitude'] = -36.47
df['elevation'] = 3160.7
df['name'] = df_sumup.reference.max() +1
df['notes'] = ''
df['error'] = np.nan
df['start_year'] = df.end_year
df[['start_date','end_date']] = np.nan

df['reference'] = df_sumup.reference.max() +1
df['reference_short'] = 'Kawakami and Iizuka (2023)'
df['reference_full'] = 'Iizuka, Y. , Matoba, S. , Yamasaki, T. , Oyabu, I. , Kadota, M. , and Aoki, T. , 2016: Glaciological and meteorological observations at the SE-Dome site, southeastern Greenland Ice Sheet. Bulletin of Glaciological Research , 34: 1–10: doi http://dx.doi.org/10.5331/bgr.15R03 . Data: Kawakami, K.; Iizuka, Y. (2023-08-11): NOAA/WDS Paleoclimatology - SE-Dome ll Ice Core, South Eastern Greenland Accumulation Rate, Melt Crust and Feature, H2O2 and Tritium Concentration, Bulk Density and Electrical Conductivity Data from 1800 to 2020 CE. [indicate subset used]. NOAA National Centers for Environmental Information. https://doi.org/10.25921/bx51-ng14.'
df['method'] = 10
df['method_str'] = 'firn or ice core, H2O2 dating'

df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)    
     
# %% GRIP 
# not added because of degraded resolution
df_meta = pd.read_csv('data/SMB data/GRIP/GRIP_gripacum.tab',
                      sep='£',engine='python',header=None)
skiprows = df_meta.index.values[df_meta[0] == '*/'][0]+1
row = df_meta.index.values[df_meta[0].str.startswith('Event')][0]

df = pd.read_csv('data/SMB data/GRIP/GRIP_gripacum.tab',
                 sep='\t', skiprows=skiprows)
# Depth ice/snow [m]	Age [ka BP]	Acc rate [g/cm**2/a]	Thickness [m]
df.columns =  ['depth', 'age','smb', 'thick']
df['start_year'] = 1950-df.age*1000
df = df.loc[df.start_year>1000,:]
df['reference_full'] = df_meta.iloc[1].values[0].split('\t')[1]
df['latitude'] = float(re.findall("(?<=LATITUDE: )\d+\.\d+", df_meta.iloc[row].values[0])[0])
df['longitude'] = -float(re.findall("(?<=LONGITUDE: -)\d+\.\d+", df_meta.iloc[row].values[0])[0])
try:
    df['elevation'] = float(re.findall("(?<=ELEVATION: )\d+\.\d+", df_meta.iloc[row].values[0])[0])
except Exception as e:
    print('Cannot find elevation')
    df['elevation'] = np.nan
df['name_str'] = df_meta.iloc[row].values[0].split('\t')[1].split(' ')[0]

df['end_year'] = df.start_year
df[['start_date', 'end_date']] = np.nan
df['notes'] = ''
df['method'] = 5
df['error'] = np.nan
df['method_str'] = 'firn or ice core'
df['reference_short'] = 'GRIP, Hammer and Dahl-Jensen (1999)'

df['name'] = resolve_name_keys(df, df_sumup)
df['reference'] = resolve_reference_keys(df, df_sumup)
print(df[['name_str', 'latitude','longitude','elevation', 'reference_short','name','reference']].drop_duplicates().values)

df.set_index('end_year').smb.plot(marker='o', ls='None')

df_sumup = pd.concat((df_sumup, df[col_needed]), ignore_index=True)        


# %% Kjær et al. 2021
print('Kjær et al. 2021')

df_meta = pd.read_csv('data/SMB data/Kjær et al. 2021/Kjaer-etal_2021_accumulation.tab',
                      sep='£',engine='python',header=None)
skiprows = df_meta.index.values[df_meta[0] == '*/'][0]+1
row = df_meta.index.values[df_meta[0].str.startswith('Event')][0]

df = pd.read_csv('data/SMB data/Kjær et al. 2021/Kjaer-etal_2021_accumulation.tab',
                 sep='\t', skiprows=skiprows)
df.columns =  ['name_str', 'Site', 'Density1m', 'Density1-2m', 'Depth rel 2019',
               'Depth rel 2018', 'Depth rel 2017', 'Depth rel 2016', 
               'Depth rel 2015', 'Depth rel 2014', '2017-2018', '2016-2017', 
               '2015-2016', '2014-2016', 'Acc snow mean', 'Acc rate std dev']

df_site_info = pd.DataFrame([['B16_LISA2019',73.935530,-37.616170],
                            ['B19_LISA2019',77.992610,-36.392220],
                            ['B22_LISA2019',79.309890,-45.673970],
                            ['NEEM_LISA2019',77.450000,-51.060000],
                            ['S4_LISA2019',77.992610,-36.392220],
                            ['S5_LISA2019',75.554930,-35.622950],
                            ['S7_LISA2019',75.629000,-35.980360]],
                            columns=['site','latitude','longitude']).set_index('site')
df['latitude'] = df_site_info.loc[df.name_str,'latitude'].values
df['longitude'] =  df_site_info.loc[df.name_str,'longitude'].values

df_stack = df[df.columns[10:-4]].stack().to_frame()
df_stack.index = df_stack.index.rename(['id','date'])
df_stack.columns = ['smb']
df_stack = df_stack.reset_index()
df_stack['name_str'] = df.loc[df_stack.id.values,'name_str'].values
df_stack['latitude'] = df.loc[df_stack.id.values,'latitude'].values
df_stack['longitude'] = df.loc[df_stack.id.values,'longitude'].values
df_stack['start_year'] = df_stack.date.str.split('-').str[0].astype(int)
df_stack['end_year'] = df_stack.date.str.split('-').str[1].astype(int)
df_stack[['start_date', 'end_date']] = np.nan
df_stack['reference_full'] = df_meta.iloc[1].values[0].split('\t')[1]
df_stack['elevation'] = np.nan
df_stack['notes'] = ''
df_stack['error'] = np.nan
df_stack['method'] = 4
df_stack['method_str'] = 'snow pits'
df_stack['reference_short'] = 'Kjær et al. (2021)'

df_stack['name'] = resolve_name_keys(df_stack, df_sumup)
df_stack['reference'] = resolve_reference_keys(df_stack, df_sumup)

for p in df_stack.name_str.unique():
    check_duplicates(df_stack.loc[df_stack.name_str==p,:], df_sumup)

df_sumup = pd.concat((df_sumup, df_stack[col_needed]), ignore_index=True)

# %% PROMICE daily ablation
plt.close('all')
path_dir = 'data/SMB data/PROMICE/'
skip_list = ['QAS_Uv3', 'QAS_Lv3', 'NUK_K','KPC_Uv3','KPC_U',
             'KAN_U', 'MIT', 'ZAK_L','ZAK_Uv3',
             'THU_U2','SWC', 'LYN_T','LYN_L',
             ]
plot = False
for f in os.listdir(path_dir):
    if f.replace('.csv','') in skip_list: 
        os.remove(path_dir + f)
        continue
    print(f)
    df = pd.read_csv(path_dir+f)
    df['time'] = pd.to_datetime(df.time)
    df = df.set_index('time')
    diff = df.z_surf_combined.diff()
    diff = diff.loc[diff<diff.quantile(0.98)]
    # diff = diff.resample('W').sum()
    
    smb = diff.copy().to_frame(name='smb') * 900/1000
    smb['year'] = smb.index.year
    smb['time'] = smb.index.values
    smb['start_date'] = smb.time - pd.Timedelta(days=1)
    smb['end_date'] = smb.time
    smb = smb.reset_index(drop=True)
    
    smb_y = smb.groupby('year').smb.sum().to_frame().reset_index(drop=True)
    smb_y['start_date'] = smb.groupby('year').time.first().values
    smb_y['end_date'] = smb.groupby('year').time.last().values
    smb_y = smb_y.reset_index(drop=True)
    
    df_new = pd.concat((smb[['start_date','end_date','smb']],
                    smb_y.loc[smb_y.start_date.dt.year>2020,
                              ['start_date','end_date','smb']]), ignore_index=True)
    
    df_new['start_year'] = smb_y.start_date.dt.year
    df_new['end_year'] = smb_y.end_date.dt.year
    df_new['latitude'] = df.gps_lat.mean()  #smb.groupby('year').time.latitude().values
    df_new['longitude'] = df.gps_lon.mean()  #smb.groupby('year').time.longitude().values
    df_new['elevation'] = df.gps_alt.mean()  # smb.groupby('year').time.elevation().values
    df_new['name_str'] = f.replace('.csv','')
    df_new['method'] = 13
    df_new['notes'] = ''
    df_new['error'] = np.nan
    df_new['method_str'] = 'pressure transducer in ablation hose'
    df_new['reference_short'] = 'PROMICE (2023)'
    df_new['reference_full'] = 'How, P.; Abermann, J.; Ahlstrøm, A.P.; Andersen, S.B.; Box, J. E.; Citterio, M.; Colgan, W.T.; Fausto. R.S.; Karlsson, N.B.; Jakobsen, J.; Langley, K.; Larsen, S.H.; Mankoff, K.D.; Pedersen, A.Ø.; Rutishauser, A.; Shield, C.L.; Solgaard, A.M.; van As, D.; Vandecrux, B.; Wright, P.J., 2022, "PROMICE and GC-Net automated weather station data in Greenland", https://doi.org/10.22008/FK2/IW73UU, GEUS Dataverse, V9'
    df_new['name'] = resolve_name_keys(df_new, df_sumup)
    df_new['reference'] = resolve_reference_keys(df_new, df_sumup)
    
    if plot:
        fig, ax = plt.subplots(2,1,sharex=True, figsize=(10,10))
        # df.z_surf_combined.plot(ax=ax[0])
        # (df.gps_alt-df.gps_alt.iloc[0]).plot(marker='o')
        smb.set_index('start_date').smb.cumsum().plot(ax=ax[0], marker='o')
        smb_y.set_index('start_date').smb.cumsum().plot(ax=ax[0], drawstyle="steps-post")
        smb_y.set_index('end_date').smb.cumsum().plot(ax=ax[0], drawstyle="steps-post")
        ax[1].set_ylabel('cumulated SMB')
        smb_y.set_index('start_date').smb.plot(ax=ax[1], drawstyle="steps-mid")
        ax[1].set_ylabel('annual SMB')
        plt.suptitle(df.site.unique()[0])

    # df_candidates = check_duplicates(df_new, df_sumup,tol = 0.1)
    df_sumup = pd.concat((df_sumup, df_new[col_needed]), ignore_index=True)


# %% checking file format
df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'longitude'] = -df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'longitude']

if df_sumup.latitude.astype(float).isnull().any():
    print('Missing latitude for')
    print(df_sumup.loc[df_sumup.latitude.astype(float).isnull(),['name_str', 'reference_short']])
    print('removing from compilation')
    print('')
    df_sumup=df_sumup.loc[df_sumup.latitude.notnull(),:]
    
if df_sumup.longitude.astype(float).isnull().any():
    print('Missing longitude for')
    print(df_sumup.loc[df_sumup.longitude.astype(float).isnull(),['name_str', 'reference_short']])
    print('removing from compilation')
    print('')
    df_sumup=df_sumup.loc[df_sumup.longitude.notnull(),:]
  
missing_method = ((df_sumup.method == -9999)|df_sumup.method.isnull())
if missing_method.any():
    print('')
    print('Missing method for')
    print(df_sumup.loc[missing_method, ['name_str', 'reference_short']].drop_duplicates)
    print('')
    df_sumup.loc[df_sumup.longitude.notnull(),'method'] = -9999

# checking inconsistent reference
df_references = df_sumup[['reference','reference_short','reference_full']].drop_duplicates()
df_references.columns = ['key','reference_short','reference']
df_references = df_references.set_index('key')
for ref in df_references.reference.loc[df_references.index.duplicated()]:
    print('\nFound reference key with multiple references:')
    print(df_references.loc[df_references.reference == ref, :].drop_duplicates().to_markdown())


print(' ======== Finished ============')
print(df_sumup.shape[0], 'observations currently in new dataset')
print(df_sumup.shape[0] - len_start, 'new observations')
# print('Checking conflicts')
# sumup_index_conflict = check_conflicts(df_sumup, df_vdx)

# print('\noverwriting conflicting data in SUMup (checked by bav)\n')
# msk = ~df_sumup.index.isin(sumup_index_conflict)
# df_sumup = pd.concat((df_sumup.loc[msk,:], df_vdx), ignore_index=True)


# looking for redundant references
# tmp = df_sumup.reference_full.unique()

print(df_sumup.shape[0], 'accumulation observations after merging from', 
      len(df_sumup.reference_full.unique()), 'sources')
print(df_sumup.loc[df_sumup.latitude>0].shape[0], 'in Greenland')
print(df_sumup.loc[df_sumup.latitude<0].shape[0], 'in Antarctica')
       
# %% writing csv files
df_ref_new = df_sumup[['reference', 'reference_full','reference_short']].drop_duplicates()
df_ref_new.columns = ['key', 'reference', 'reference_short']
# df_ref_new['key'] = np.arange(1,len(df_ref_new)+1)
df_ref_new = df_ref_new.set_index('key')
df_ref_new.to_csv('SUMup 2023 beta/SUMup_2023_SMB_csv/SUMup_2023_SMB_references.tsv', sep='\t')
# df_sumup.reference = df_ref_new.reset_index().set_index('reference').loc[df_sumup.reference_full].values

df_sumup.loc[df_sumup.method_str =='firn core', 'method_str'] = 'firn or ice core'
df_method_new = pd.DataFrame(df_sumup.method_str.unique())
df_method_new.columns = ['method']
df_method_new.index = df_method_new.index+1
df_method_new.index.name = 'key'
df_method_new.to_csv('SUMup 2023 beta/SUMup_2023_SMB_csv/SUMup_2023_SMB_methods.tsv', sep='\t')
df_sumup.method = df_method_new.reset_index().set_index('method').loc[df_sumup.method_str].values

df_name_new = pd.DataFrame(df_sumup.name_str.unique())
df_name_new.columns = ['name']
df_name_new['key'] = np.arange(1,len(df_name_new)+1)
df_name_new = df_name_new.set_index('key')
df_name_new.to_csv('SUMup 2023 beta/SUMup_2023_SMB_csv/SUMup_2023_SMB_names.tsv', sep='\t')
df_sumup['name'] = df_name_new.reset_index().set_index('name').loc[df_sumup.name_str].values


df_sumup['latitude'] = df_sumup.latitude.astype(float).round(6)
df_sumup['longitude'] = df_sumup.longitude.astype(float).round(6)
df_sumup['smb'] = df_sumup.smb.astype(float).round(4)
df_sumup.loc[df_sumup['error']==0,'error'] = np.nan
df_sumup['error'] = df_sumup.error.astype(float).round(4)

df_sumup.loc[df_sumup.elevation.isnull(), 'elevation'] = -9999
df_sumup['elevation'] = df_sumup.elevation.round(0).astype(int)
df_sumup['elevation'] = df_sumup['elevation'].astype(str).replace('-9999','')

df_sumup['start_date'] = pd.to_datetime(df_sumup.start_date)
df_sumup['end_date'] = pd.to_datetime(df_sumup.end_date)

msk = df_sumup.start_year.isnull() & df_sumup.start_date.notnull()
df_sumup.loc[msk, 'start_year'] = df_sumup.loc[msk, 'start_date'].dt.year
df_sumup.loc[df_sumup.start_year.isnull(), 'start_year'] = -9999
df_sumup['start_year'] = df_sumup.start_year.astype(int)
df_sumup['start_year'] = df_sumup['start_year'].astype(str).replace('-9999','')

msk = df_sumup.end_year.isnull() & df_sumup.end_date.notnull()
df_sumup.loc[msk, 'end_year'] = df_sumup.loc[msk, 'end_date'].dt.year
df_sumup.loc[df_sumup.end_year.isnull(), 'end_year'] = -9999
df_sumup['end_year'] = df_sumup.end_year.round(0).astype(int)
df_sumup['end_year'] = df_sumup['end_year'].astype(str).replace('-9999','')

df_sumup['method'] = df_sumup['method'].astype(str).replace('-9999','')

df_sumup = df_sumup.rename(columns={'name':'name_key',
                                'name_str':'name',
                                'method':'method_key',
                                'method_str':'method',
                                'reference':'reference_key',
                                'reference_full':'reference'})
    
df_sumup.loc[df_sumup.latitude>0, ['name_key', 'reference_key', 'method_key', 'start_date', 'end_date',
                                   'start_year', 'end_year', 'latitude', 'longitude', 
                                   'elevation',  'smb',  'error', 'notes']
             ].to_csv('SUMup 2023 beta/SUMup_2023_SMB_csv/SUMup_2023_SMB_greenland.csv',
                            index=None)
df_sumup.loc[df_sumup.latitude<0, ['name_key', 'reference_key', 'method_key', 'start_date', 'end_date',
                                   'start_year', 'end_year','latitude', 'longitude',  
                                   'elevation',  'smb',  'error', 'notes']
             ].to_csv('SUMup 2023 beta/SUMup_2023_SMB_csv/SUMup_2023_SMB_antarctica.csv',
                            index=None)
import shutil
shutil.make_archive('SUMup 2023 beta/SUMup_2023_SMB_csv',
                    'zip', 'SUMup 2023 beta/SUMup_2023_SMB_csv')
# %% netcdf format
import xarray as xr
df_sumup[['start_year', 'end_year', 'method_key', 'elevation']] = \
    df_sumup[['start_year', 'end_year', 'method_key', 'elevation']].replace('','-9999').astype(int)

def write_netcdf(df_sumup, filename):
    df_new = df_sumup.copy()
    df_new['start_date'] = pd.to_datetime(df_new.start_date).dt.tz_localize(None)
    df_new['end_date'] = pd.to_datetime(df_new.end_date).dt.tz_localize(None)

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
    
    ds_sumup.start_date.encoding['units'] = 'days since 1900-01-01'
    ds_sumup.end_date.encoding['units'] = 'days since 1900-01-01'
    
    # attributes
    df_attr = pd.read_csv('doc/attributes_smb.csv',
                          skipinitialspace=True,
                          comment='#').set_index('var')
    for v in df_attr.index:
        for c in df_attr.columns:
            if v in ds_sumup.keys():
                ds_sumup[v].attrs[c] = df_attr.loc[v,c]
            if v in ds_meta.keys():
                ds_meta[v].attrs[c] = df_attr.loc[v,c]
    if ds_sumup.latitude.isel(measurement_id=0)>0:
        ds_sumup.attrs['title'] = 'SUMup SMB dataset for the Greenland ice sheet (2023 release)'
    else:
        ds_sumup.attrs['title'] = 'SUMup SMB dataset for the Antarctica ice sheet (2023 release)'
    ds_sumup.attrs['contact'] = 'Baptiste Vandecrux'
    ds_sumup.attrs['email'] = 'bav@geus.dk'
    ds_sumup.attrs['production date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
       
    float_encoding = {"dtype": "float32", "zlib": True,"complevel": 9}
    int_encoding = {"dtype": "int32", "_FillValue":-9999, "zlib": True,"complevel": 9}
    
    ds_sumup[['name_key', 'reference_key', 'method_key', 'start_date', 'end_date',
              'start_year', 'end_year','latitude', 'longitude', 'elevation',  'smb',  
              'error']].to_netcdf(filename, 
                                  group='DATA',
                                  encoding={
                                     "smb": float_encoding |{'least_significant_digit':4},
                                     "start_date": int_encoding,
                                     "end_date": int_encoding,
                                     "start_year": int_encoding,
                                     "end_year": int_encoding,
                                     "error": float_encoding|{'least_significant_digit':4},
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
    
write_netcdf(df_sumup.loc[df_sumup.latitude>0, :], 'SUMup 2023 beta/SUMup_2023_SMB_greenland.nc')
write_netcdf(df_sumup.loc[df_sumup.latitude<0, :], 'SUMup 2023 beta/SUMup_2023_SMB_antarctica.nc')
#%% updating ReadMe file
    
df_meta = pd.DataFrame()
df_meta['total'] = [df_sumup.shape[0]]
df_meta['added'] = df_sumup.shape[0]-len_sumup_2022
df_meta['nr_references'] = str(len(df_sumup.reference.unique()))
df_meta['greenland'] = df_sumup.loc[df_sumup.latitude>0].shape[0]
df_meta['antarctica'] = df_sumup.loc[df_sumup.latitude<0].shape[0]
df_meta.index.name='index'
df_meta.to_csv('doc/ReadMe_2023_src/tables/SMB_meta.csv')

print('{:,.0f}'.format(df_sumup.shape[0]).replace(',',' ') +\
      ' SMB observations in SUMup 2023')
print('{:,.0f}'.format(df_sumup.shape[0]-len_sumup_2022).replace(',',' ') +\
      ' more than in SUMup 2022')
print('from '+ str(len(df_sumup.reference_short.unique())) + ' sources')
print('representing '+ str(len(df_sumup.reference.unique()))+' references')

print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude>0].shape[0]).replace(',',' ')+' observations in Greenland')
print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude<0].shape[0]).replace(',',' ')+' observations in Antarctica')

plot_dataset_composition(df_sumup.loc[df_sumup.latitude>0], 
        'doc/ReadMe_2023_src/figures/SMB_dataset_composition_greenland.png')
plot_map(df_sumup.loc[df_sumup.latitude>0,['latitude','longitude']].drop_duplicates(),
         'doc/ReadMe_2023_src/figures/SMB_map_greenland.png', 
         area='greenland')

plot_dataset_composition(df_sumup.loc[df_sumup.latitude<0],
        'doc/ReadMe_2023_src/figures/SMB_dataset_composition_antarctica.png')

plot_map(df_sumup.loc[df_sumup.latitude<0,['latitude','longitude']].drop_duplicates(),
         'doc/ReadMe_2023_src/figures/SMB_map_antarctica.png', 
         area='antarctica')


print_table_dataset_composition(df_sumup.loc[df_sumup.latitude>0]).to_csv('doc/ReadMe_2023_src/tables/composition_SMB_greenland.csv',index=None)

print_table_dataset_composition(df_sumup.loc[df_sumup.latitude<0]).to_csv('doc/ReadMe_2023_src/tables/composition_SMB_antarctica.csv',index=None)

print('writing out measurement locations')
print_location_file(df_sumup.loc[df_sumup.latitude>0,:], 
                    'doc/GIS/SUMup_2023_smb_location_greenland.csv')

print_location_file(df_sumup.loc[df_sumup.latitude<0, :],
                    'doc/GIS/SUMup_2023_smb_location_antarctica.csv')


