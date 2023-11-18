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
import os
import matplotlib.pyplot as plt
from sumup_lib import *
plt.close('all')
# loading and preparing the data
try:
    df_sumup = pd.read_csv('data/SUMup 2022/SUMup_density_2022.csv')
except:
    print('Downloading SUMup 2022 density file')
    url = 'https://cn.dataone.org/cn/v2/resolve/urn:uuid:af7b851c-18ec-45a3-a5c8-8fcb98f62665'
    r = requests.get(url, allow_redirects=True)
    open('data/SUMup 2022/SUMup_density_2022.csv', 'wb').write(r.content)    
    df_sumup = pd.read_csv('data/SUMup 2022/SUMup_density_2022.csv')

df_sumup.columns = df_sumup.columns.str.lower()
df_sumup = df_sumup.rename(columns={'citation': 'reference'})

# correcting an error in SUMup 2022:
print(df_sumup.loc[df_sumup.profile == 191, ['profile','latitude', 
                                             'longitude','elevation']].iloc[0, :])
df_sumup.loc[df_sumup.profile == 191, ['profile_name','latitude','longitude',
                                       'elevation']] = '00-20', 76.74, -66.18, 1295
print(df_sumup.loc[df_sumup.profile == 191, ['profile','latitude', 
                                             'longitude','elevation']].iloc[0, :])

# fixing profile from Miege 2020 on the Wilkins IS
df_sumup.loc[df_sumup.profile==1943,'profile'] = 1942

# fixing method for Niwano et al. (2020)
for p in [1969, 1970, 1971, 1972]:
    df_sumup.loc[df_sumup.profile==p,'method'] = 3

if df_sumup.latitude.isnull().sum() | (df_sumup.latitude==-9999).sum(): print(wtf)

df_ref = pd.read_csv('data/SUMup 2022/SUMup_density_references_2022.txt', sep='eirj', engine='python', header=None)
df_ref.columns = ['reference']
df_ref['key'] = np.arange(1,len(df_ref)+1)
df_ref = df_ref.set_index('key')

df_names = pd.read_csv('data/SUMup 2022/SUMup_density_2022_profile_names.csv')
df_names = df_names.set_index('key')
for ind in df_names.index:
    df_sumup.loc[df_sumup.profile==ind,'profile_name'] = df_names.loc[ind,'name']

df_methods = pd.read_csv('data/SUMup 2022/SUMup_density_methods_2022.txt', sep='eirj', engine='python', header=None)
df_methods.columns = ['method']
df_methods['key'] = np.arange(1,len(df_methods)+1)
df_methods = df_methods.set_index('key')

# df_sumup['name_str'] = df_names.loc[df_sumup.name, 'name'].values
df_sumup['reference_full'] = df_ref.loc[df_sumup.reference, 'reference'].values
df_sumup['method_str'] = df_methods.loc[df_sumup.method, 'method'].values

short_ref = parse_short_reference(df_sumup).set_index('reference')

for ref, s in zip(short_ref.index, short_ref.reference_short):
    if s.startswith('Smeets et al. (2016'): short_ref.loc[ref, 'reference_short'] = 'Smeets et al. (2016a,b,c,d)'  
    if s.startswith('Wilhelms et al. (2000'): short_ref.loc[ref, 'reference_short'] = 'Wilhelms et al. (2000a,b,c,d)'  
    if s.startswith('Miller et al. (2000'): short_ref.loc[ref, 'reference_short'] = 'Miller and Schwager (2000a,b)'  
    if s.startswith('Graf et al. (2002'): short_ref.loc[ref, 'reference_short'] = 'Graf et al. (2002a,b,c,d,e,f,g)'  
    if s.startswith('Bolzan et al. (1999'): short_ref.loc[ref, 'reference_short'] = 'Bolzan and Strobel (1999a,b,c,d,e,f,g,h,i,j,k,l,m,n,o)'   
    if s.startswith('Bolzan et al. (2001'): short_ref.loc[ref, 'reference_short'] = 'Bolzan and Strobel (2001a,b)'  
    if s.startswith('Graf et al. (2006'): short_ref.loc[ref, 'reference_short'] = 'Graf and Oerter (2006a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z)'   
    if s.startswith('Graf et al. (1988'): short_ref.loc[ref, 'reference_short'] = 'Graf et al. (1988a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q)'
    if s.startswith('Graf et al. (1999'): short_ref.loc[ref, 'reference_short'] = 'Graf et al. (1999a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)'
    if s.startswith('Graf et al. (2002'):  short_ref.loc[ref, 'reference_short'] = 'Graf et al. (2002a,b,c,d,e,f,g,h,i,j,k,l,m,n,o)'
    if s.startswith('Oerter et al. (1999'): short_ref.loc[ref, 'reference_short'] = 'Oerter et al. (1999a,b,c,d,e,f,g,h)'
    if s.startswith('Oerter et al. (2000'): short_ref.loc[ref, 'reference_short'] = 'Oerter et al. (2000a,b,c,d,e,f,g,h,i)'
    if s.startswith('Oerter et al. (2008'): short_ref.loc[ref, 'reference_short'] = 'Oerter et al. (2008a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)'
    if s == 'Mayewski et al. (2016)': short_ref.loc[ref, 'reference_short'] = 'Mayewski and Whitlow (2016)'
    if s == 'Dibb et al. (2004)': short_ref.loc[ref, 'reference_short'] = 'Dibb and Fahnestock (2004)'
        
df_sumup['reference_short'] = short_ref.loc[df_sumup.reference_full].reference_short.values

len_sumup_2022 = df_sumup.shape[0]
print('{:,.0f}'.format(df_sumup.shape[0]).replace(',',' '),
      'density observations in SUMup 2022')
print('from', len(df_sumup.reference_short.unique()), 'sources')
print('representing', len(df_sumup.reference.unique()), 'references')
print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude>0].shape[0]).replace(',',' '), 'in Greenland')
print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude<0].shape[0]).replace(',',' '), 'in Antarctica')


# Variables needed
necessary_variables = ['profile','profile_name', 'reference', 'reference_short', 
                       'reference_full', 'method', 'date', 'timestamp', 'latitude', 
                       'longitude', 'elevation', 'start_depth', 'stop_depth',
                       'midpoint', 'density', 'error']

# %% 2023 additions (Greenland profiles) 
l1 = df_sumup.shape[0]
df_sumup = add_Greenland_profiles(df_sumup, necessary_variables, df_ref, short_ref)
l2 = df_sumup.shape[0]
print('')
print('Added', l2-l1, 'measurements')

# %% Wilhelms 2004
print('adding Wilhelms 2004')

f = "data/density data/Wilhelms2004/DML94C07_38_density.tab"
df_meta = pd.read_csv(f, sep='£',
                      engine='python', header=None)
skiprows = df_meta.index.values[df_meta[0] == '*/'][0]+1
row = df_meta.index.values[df_meta[0].str.startswith('Event')][0]

df = pd.read_csv(f, sep='\t', skiprows=skiprows)
df.columns =  ['midpoint', 'density','density_std']
df['start_depth'] = df.midpoint - df.midpoint.diff().round(4).median()
df['stop_depth'] = df.midpoint + df.midpoint.diff().round(4).median()

df['reference_full'] = df_meta.iloc[1].values[0].split('\t')[1]
df['latitude'] = float(re.findall("(?<=LATITUDE: )[-+]?\d+\.\d+", df_meta.iloc[row].values[0])[0])
df['longitude'] = float(re.findall("(?<=LONGITUDE: )[-+]?\d+\.\d+", df_meta.iloc[row].values[0])[0])
df['elevation'] = float(re.findall("(?<=ELEVATION: )\d+\.\d+", df_meta.iloc[row].values[0])[0])

df['timestamp'] = re.findall("(\d{4}-\d{2}-\d{2})",
                             df_meta.iloc[row].values[0])[0]
df['date'] = df['timestamp'].str.replace('-','').astype(int)
df['profile_name'] = df_meta.iloc[row].values[0].split('\t')[1].split(' ')[0]

df['method'] = 11
df['error'] = np.nan
df['profile'] = df_sumup.profile.max()+1
df['reference'] = df_sumup.reference.max()+1
df['reference_short'] = "Wilhelms (2007a,b)"
# print(df[['profile_name','date', 'latitude','longitude','elevation', 'reference_short']].drop_duplicates().values)

sumup_index_conflict = check_conflicts(df_sumup, df,
                                        var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% Akers 2022 core
print('loading Akers 2022 core')
df = pd.read_csv('data/density data/Akers2022/ABN1314-103_density_SMB.tab',
                 sep='\t', skiprows=24)

df = df.iloc[:, [1, 2, 3, 8]]

df.columns = ['start_depth', 'stop_depth','midpoint', 'density']
df['reference_full'] = 'Akers, Pete D; Savarino, Joël; Caillon, Nicolas; Servettaz, Aymeric P M; Le Meur, Emmanuel; Magand, Olivier; Agosta, Cécile; Crockford, Peter; Kobayashi, Kanon; Hattori, Shohei; Curran, Mark; van Ommen, Tas D; Jong, Lenneke; Roberts, Jason L; Martins, Jean (2022): Ice density-based surface mass balance from the ABN1314-103 ice core, Aurora Basin North, Antarctica. PANGAEA, https://doi.org/10.1594/PANGAEA.941489'
df['reference_short'] = 'Akers et al. (2022)'
df['reference'] = df_sumup.reference.max() + 1
df['profile'] = df_sumup.profile.max() + 1
df['profile_name'] = 'ABN1314-103'
df['latitude'] = -71.170000
df['longitude'] = 111.370000
df['elevation'] = 2679
df['method'] = 4
df['date'] = 20140107
df['timestamp'] = pd.to_datetime('2014-01-07')
df['error'] = -9999

sumup_index_conflict = check_conflicts(df_sumup, df,
                                        var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

#%% Stevens 2023 USP50
print('loading Stevens 2023 USP50')

for k, f in enumerate(os.listdir("data/density data/Stevens2023/")):
    if 'README' in f:
        continue
    if 'datarelease' in f:
        continue
    df = pd.read_csv("data/density data/Stevens2023/"+f)

    df.columns = ['start_depth','density']

    if '106' in f:
        df['density'] = df['density']*1000
    df['stop_depth']=df['start_depth']+df['start_depth'].diff()
    
    df['midpoint'] = df['start_depth'] + (df['stop_depth'] - df['start_depth'])/2
   
    df['reference_full'] = "Stevens, C., Conway, H., Fudge, T. J., Koutnik, M., Lilien, D., & Waddington, E. D. (2023) Firn density and compaction rates 50km upstream of South Pole U.S. Antarctic Program (USAP) Data Center. doi: https://doi.org/10.15784/601680. "
    df['latitude'] = -89.54
    df['longitude'] =  137.04
    df['elevation'] = np.nan
    df['profile_name'] = f[:-4]
    df['timestamp'] ='2016-12-01'
    df['date'] = df['timestamp'].str.replace('-','').astype(int)
    df['method'] = 4
    df['error'] = np.nan
    df['profile'] = df_sumup.profile.max()+1
    df['reference'] = df_sumup.reference.max()+1
    df['reference_short'] = "Stevens et al. (2023)"
    print(df[['profile_name','date', 'latitude','longitude','elevation', 'reference_short']].drop_duplicates().values)
    sumup_index_conflict = check_conflicts(df_sumup, df,
        var=['profile', 'profile_name', 'date', 'start_depth', 'density'])

    df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% Albert 2015 WAIS core
print('loading Albert 2015 WAIS core')
df = pd.read_excel('data/density data/Albert2015/WAISDivide_FirnProperties.xlsx',
                   header=None)

df = df.iloc[11:, [1, 2, 3, 4]]

df.columns = ['start_depth', 'midpoint','thickness', 'density']
df['density'] = df.density*1000
df['stop_depth'] = df.start_depth + df.thickness/1000
df['reference_full'] = 'Gregory, S. A., Albert, M. R., and Baker, I.: Impact of physical properties and accumulation rate on pore close-off in layered firn, The Cryosphere, 8, 91–105, https://doi.org/10.5194/tc-8-91-2014, 2014. Data: Mary, A. 2015. Firn Permeability and Density at WAIS Divide. Boulder, Colorado USA: National Snow and Ice Data Center. http://dx.doi.org/10.7265/N57942NT'
df['reference_short'] = 'Gregory et al. (2014), Albert (2015)'
df['reference'] = df_sumup.reference.max() + 1
df['profile'] = df_sumup.profile.max() + 1
df['profile_name'] = 'WAIS_WDC05C'
df['latitude'] = -79.46300
df['longitude'] = -112.12317
df['elevation'] = np.nan
df['method'] = 4
df['date'] = 20051201
df['timestamp'] = pd.to_datetime('2005-12-01')
df['error'] = -9999
df.plot(x='midpoint',y='density')
sumup_index_conflict = check_conflicts(df_sumup, df,
                           var=['profile', 'profile_name', 'date',
                                'start_depth', 'density'])
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)


# %% Fourteau2019b
print('loading Fourteau2019b')

for f in os.listdir('data/density data/Fourteau2019b'):
    if 'summary' in f: continue
    df_meta = pd.read_csv('data/density data/Fourteau2019b/'+f, sep='£', engine='python', header=None)
    skiprows = df_meta.index.values[df_meta[0] == '*/'][0]+1
    row = df_meta.index.values[df_meta[0].str.startswith('Event')][0]
    
    df = pd.read_csv('data/density data/Fourteau2019b/'+f, sep='\t', skiprows=skiprows)
    df.columns =  ['midpoint', 'poros_open','poros_clos','poros_frac','poros_vol','vol','density','dens_rel']
    df['density'] = df.density*1000
    half_thick = (df.midpoint.values[1:] - df.midpoint.values[:-1])/2
    half_thick = np.append(half_thick, half_thick[-1])
    df['start_depth'] = df.midpoint - half_thick
    df['stop_depth'] = df.midpoint + half_thick
    
    df['reference_full'] = 'Fourteau, K., Arnaud, L., Faïn, X., Martinerie, P., Etheridge, D. M., Lipenkov, V., and Barnola, J.-M.: Historical porosity data in polar firn, Earth Syst. Sci. Data, 12, 1171–1177, https://doi.org/10.5194/essd-12-1171-2020, 2020. Data: '+ df_meta.iloc[1].values[0].split('\t')[1]
    df['latitude'] = float(re.findall("(?<=LATITUDE: )[-+]?\d+\.\d+", df_meta.iloc[row].values[0])[0])
    df['longitude'] = float(re.findall("(?<=LONGITUDE: )[-+]?\d+\.\d+", df_meta.iloc[row].values[0])[0])
    df['elevation'] = np.nan
    
    if 'Vostok' in f:
        df['profile_name'] = 'Vostok_BH3' 
        df['timestamp'] = '1991-12-20'
    if 'Summit' in f:
        df['profile_name'] = 'Summit_EUROCORE'
        df['timestamp'] = '1989-06-01'
    if 'DE08' in f:
        df['profile_name'] = 'DE08-2'
        df['timestamp'] = '1992-12-20'
    
    df['date'] = df['timestamp'].str.replace('-','').astype(int)
    df['method'] = 4
    df['error'] = np.nan
    df['profile'] = df_sumup.profile.max()+1
    df['reference'] = df_sumup.reference.max()+1
    df['reference_short'] = "Fourteau (2019a,b,c)"
    print(df[['profile_name','date', 'latitude','longitude','elevation', 'reference_short']].drop_duplicates().values)

    sumup_index_conflict = check_conflicts(df_sumup, df,
                                            var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
    df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% Dome A CHINARE core
print('loading Dome A CHINARE core')
df = pd.read_excel('data/density data/Dome A/Dome-A-2005-density.xlsx',
                   header=None)

df = df.iloc[5:, :]
df.columns = ['density', '?','stop_depth']
df['stop_depth'] = df.stop_depth/100
df['start_depth'] = pd.concat((pd.Series([0]),
                               df.stop_depth.iloc[:-1])).reset_index(drop=True).values
df['midpoint'] = df['start_depth'] + (df['stop_depth'] - df['start_depth'])/2
df['density'] = df.density*1000

df['reference_full'] = 'Cunde, X., Yuansheng, L., Allison, I., Shugui, H., Dreyfus, G., Barnola, J., . . . Kameda, T. (2008). Surface characteristics at Dome A, Antarctica: First measurements and a guide to future ice-coring sites. Annals of Glaciology, 48, 82-87. doi:10.3189/172756408784700653'
df['reference_short'] = 'Cunde et al. (2008)'
df['reference'] = df_sumup.reference.max() + 1
df['profile'] = df_sumup.profile.max() + 1
df['profile_name'] = 'Dome A CHINARE-21'

df['latitude'] = -80.3671111
df['longitude'] = 77.3728611111111
df['elevation'] = 4093
df['method'] = 4
df['date'] = 20050101
df['timestamp'] = pd.to_datetime('2005-01-01')
df['error'] = -9999
# df.plot(x='midpoint',y='density')
sumup_index_conflict = check_conflicts(df_sumup, df,
                           var=['profile', 'profile_name', 'date',
                                'start_depth', 'density'])
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
# %% Taldice
print('loading Taldice core')
df = pd.read_excel('data/density data/TALDICE/TALDICE_density_profile.xls',
                   header=None)

df = df.iloc[9:, [1, 5]]
df.columns = ['midpoint','density']
df['density'] = df.density*1000
intervals = df.midpoint.diff().fillna(1)  # Fill NaN for the first element with 0
df['start_depth'] = df.midpoint - intervals / 2
df['stop_depth'] = df.midpoint + intervals / 2

df['midpoint'] = df['start_depth'] + (df['stop_depth'] - df['start_depth'])/2

df['reference_full'] = 'TalDIce ice core'
df['reference_short'] = 'TalDIce ice core'
df['reference'] = df_sumup.reference.max() + 1
df['profile'] = df_sumup.profile.max() + 1
df['profile_name'] = 'TalDIce ice core'

df['latitude'] =  -72.8166667
df['longitude'] = 159.18333333333334
df['elevation'] = 2315
df['method'] = 4
df['date'] = 19961101
df['timestamp'] = pd.to_datetime('1996-11-01')
df['error'] = -9999
# df.plot(x='midpoint',y='density')
sumup_index_conflict = check_conflicts(df_sumup, df,
                           var=['profile', 'profile_name', 'date',
                                'start_depth', 'density'])
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% Wais data
ind_ref = df_sumup.reference.max() + 1

print('loading WAIS data')
dir_path = 'data/density data/WAIS data from Christo Buizert/'
for f in os.listdir(dir_path):
    if 'Sower' not in f : continue
    df = pd.read_excel(dir_path+f,  header=None)
    year, alt = re.findall(r'\d+', df.iloc[3,0])
    numbers = re.findall(r'\d+', df.iloc[2,0])
    if 'WDC05E' in f: lat= -(79+27.777/60); lon = -(112+07.506/60)
    if 'WDC05C' in f: lat= -(79+27.777/60); lon = -(112+07.362/60)
    if 'WDC05A' in f: lat= -(79+27.777/60); lon = -(112+07.506/60)    
    
    df = df.iloc[6:, :]
    
    if 'WDC05E' in f:
        df.columns = ['start_depth', 'stop_depth',  'density', 'midpoint', '?']
    else:
        df.columns = ['start_depth', 'stop_depth', 'midpoint', 'density', '?','length']
    df['density'] = df.density*1000
    
    df['reference_full'] = 'WAIS cores, T. Sowers, C. Buizert, personal communication'
    df['reference_short'] = 'WAIS cores'
    df['reference'] = ind_ref
    df['profile'] = df_sumup.profile.max() + 1
    df['profile_name'] = f.split('_')[0]
    
    df['latitude'] = lat
    df['longitude'] = lon
    df['elevation'] = alt
    df['method'] = 4
    df['date'] = 20050101
    df['timestamp'] = pd.to_datetime('2005-01-01')
    df['error'] = -9999
    # df.plot(x='midpoint',y='density')
    print(df[['profile_name','date', 'latitude','longitude','elevation', 'reference_short']].drop_duplicates().values)

    sumup_index_conflict = check_conflicts(df_sumup, df,
                               var=['profile', 'profile_name', 'date',
                                    'start_depth', 'density'])
    df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

# %% SPICE data
ind_ref = df_sumup.reference.max() + 1

print('loading SPICE data')
dir_path = 'data/density data/SPICE data from Christo Buizert/'
df = pd.read_excel(dir_path+'SPICE_density_SowersBuizert_2015.xlsx',  header=None)

df1 = df.iloc[3:, [1,4,5]]
df1.columns = ['l','midpoint','density']
df1['profile_name'] = 'SP 15 FC1'
df1['date'] = 20151101
df1['timestamp'] = pd.to_datetime('2015-11-01')
df1['profile'] = df_sumup.profile.max() + 1
df1 = df1.loc[df1.density.notnull()]

df2 = df.iloc[3:, [8, 11,12]]
df2.columns = ['l','midpoint','density']
df2['profile_name'] = 'SP 15 FC2'
df2['date'] = 20151201
df2['timestamp'] = pd.to_datetime('2015-12-01')
df2['profile'] = df_sumup.profile.max() + 2

df = pd.concat((df1,df2), ignore_index=True)
df['start_depth'] = df.midpoint - df.l/100 / 2
df['stop_depth'] = df.midpoint + df.l/100 / 2
df['density'] = df.density*1000

df['reference_full'] = 'SPICE cores, T. Sowers, C. Buizert, personal communication'
df['reference_short'] = 'SPICE cores'
df['reference'] = ind_ref

df['latitude'] = -90
df['longitude'] = 0
df['elevation'] = np.nan
df['method'] = 4
df['error'] = -9999

df[['midpoint','density','start_depth','stop_depth']] = df[['midpoint','density','start_depth','stop_depth']].astype(float)

# plt.figure()
# df.plot(x='midpoint',y='density')
print(df[['profile_name','date', 'latitude','longitude','elevation', 'reference_short']].drop_duplicates().values)

sumup_index_conflict = check_conflicts(df_sumup, df,
                           var=['profile', 'profile_name', 'date',
                                'start_depth', 'density'])
df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)


# %% Quality tests (units and missing fields)
# tests to know if some units have been missinterpreted
df_ref_short = df_sumup[['reference','reference_short']].drop_duplicates().set_index('reference')

wrong_density_unit = (df_sumup.groupby('reference').density.mean()<1)
assert wrong_density_unit.sum()==0, print('Ref. that have wrong density unit:', df_ref_short.loc[wrong_density_unit.index[wrong_density_unit]])

wrong_density_unit = (df_sumup.groupby('reference').density.mean()>2000)
assert wrong_density_unit.sum()==0, print('Ref. that have wrong density unit:', df_ref_short.loc[wrong_density_unit.index[wrong_density_unit]])
    
# calculating midpoint for the reference that don't have it
print(df_sumup.midpoint.isnull().sum(), 'measurements missing midpoint')
df_sumup.loc[df_sumup.midpoint.isnull(), 'midpoint'] = \
    df_sumup.loc[df_sumup.midpoint.isnull(), 'start_depth'] \
        + (df_sumup.loc[df_sumup.midpoint.isnull(), 'start_depth'] \
           -df_sumup.loc[df_sumup.midpoint.isnull(), 'stop_depth'])/2

# looking for depth added as cm instead of m
df_p = df_sumup[['profile','profile_name']].drop_duplicates().set_index('profile')
profiles = df_sumup.loc[(df_sumup.midpoint > 100) & (df_sumup.density < 700),'profile'].drop_duplicates().values
plt.close('all')
for p in profiles:
    print(p)
    plt.figure()
    df_sumup.loc[df_sumup.profile==p,:].plot(ax=plt.gca(),
                                             x='density',
                                             y='midpoint',
                                             marker='o',
                                             label=df_p.loc[p].values[0])
 
# missing latitude
assert not (df_sumup.latitude.isnull().sum() | (df_sumup.latitude==-9999).sum()), "some profile missing latitude"


# positive longitudes in greenland
if len(df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'profile_name'])>0:
    print(df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'longitude'].drop_duplicates())
assert len(df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'profile_name'])==0, "some Greenland measurement has positive longitudes"
# df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'longitude'] = -df_sumup.loc[(df_sumup.latitude>0)&(df_sumup.longitude>0), 'longitude']

# missing profile keys
assert not df_sumup.profile.isnull().any(), "some profile have NaN as profile key"
if df_sumup.profile.isnull().any():
    print(df_sumup.loc[df_sumup.profile.isnull(), ['profile', 'profile_name','reference_short']].drop_duplicates())
    
# missing time stamp
if df_sumup['timestamp'].isnull().any():
    print('Missing time stamp for:')
    print(df_sumup.loc[df_sumup['timestamp'].isnull()|(df_sumup['timestamp']=='Na'), ['profile_name','reference_short']].drop_duplicates())
    print('Ignoring entry')
    df_sumup = df_sumup.loc[
        df_sumup['timestamp'].notnull()&(df_sumup['timestamp']!='Na'),:]
    
# checking duplicate reference
tmp = df_sumup[['reference','reference_full']].drop_duplicates()
if tmp.reference.duplicated().any():
    print('\n====> Found two references for same reference key')
    dup_ref = tmp.loc[tmp.reference.duplicated()]
    for ref in dup_ref.reference.values:
        doubled_ref = df_sumup.loc[df_sumup.reference == ref,
                           ['reference','reference_full']].drop_duplicates()
        if doubled_ref.iloc[0,1].replace(' ','').lower() == doubled_ref.iloc[1,1].replace(' ','').lower():
            df_sumup.loc[df_sumup.reference == ref, 'reference_full'] = doubled_ref.iloc[0,1]
            print('Merging\n', doubled_ref.iloc[1,1],'\ninto\n', doubled_ref.iloc[0,1],'\n')
        else:
            print(wtf)
df_meta = df_sumup[['profile','method','method_str']].drop_duplicates()            
for ind in df_meta.index[df_meta.index.duplicated()]:
    if len(df_meta.loc[ind,'method_str'])>1:
        print(wtf)
        print('\n> found profile with multiple methods')
        print(df_meta.loc[ind,['profile_name', 'method_str', 'reference_short']])
        print('renaming method for this profile')
        df_meta.loc[ind, 'method_str'] = ' or '.join(df_meta.loc[ind, 'method_str'].tolist())
print('=== Finished ===')
# %% renaming variables

df_sumup = df_sumup.rename(columns={'profile':'profile_key',
                                    'profile_name':'profile',
                                    'method':'method_key',
                                    'method_str':'method',
                                    'reference':'reference_key',
                                    'reference_full':'reference'})

# %% writing CSV file
# saving new metadata files
df_sumup.reference_key = df_sumup.reference_key.astype(int)
df_ref_new = df_sumup[['reference_key','reference','reference_short']].drop_duplicates()
df_ref_new.columns = ['key', 'reference','reference_short']
df_ref_new = df_ref_new.set_index('key').sort_index()
df_ref_new.to_csv('SUMup 2023 beta/SUMup_2023_density_csv/SUMup_2023_density_references.tsv', sep='\t')

df_sumup.loc[df_sumup.method_key.isnull(),'method'] = 'NA'
df_sumup.loc[df_sumup.method_key == -9999,'method'] = 'NA'
df_sumup.loc[df_sumup.method_key.isnull(),'method_key'] = -9999
df_method_new = df_sumup[['method_key','method']].drop_duplicates()
df_method_new.columns = ['key', 'method']
df_method_new.loc[-9999,'method'] = 'Not available'
df_method_new = df_method_new.dropna()
df_method_new = df_method_new.set_index('key').sort_index()
df_method_new.index = df_method_new.index.astype(int)
df_method_new.to_csv('SUMup 2023 beta/SUMup_2023_density_csv/SUMup_2023_density_methods.tsv', sep='\t')

df_sumup.profile_key = df_sumup.profile_key.astype(int)
df_sumup.loc[df_sumup.profile.isnull(),'profile'] = 'NA'
df_profiles = df_sumup[['profile_key','profile']].drop_duplicates()
df_profiles.columns = ['key', 'profile']
df_profiles = df_profiles.set_index('key').sort_index()
df_profiles.to_csv('SUMup 2023 beta/SUMup_2023_density_csv/SUMup_2023_density_profile_names.tsv', sep='\t')

# removing time for now, exception should be fixed at some point
df_sumup['timestamp'] = df_sumup.timestamp.astype(str).str.split(' ').str[0].str.split('T').str[0]
try:
    df_sumup['timestamp'] = pd.to_datetime(df_sumup.timestamp, format='mixed')
except:
    df_sumup['timestamp'] = pd.to_datetime(df_sumup.timestamp, utc=True)

# all longitude should be ranging from -180 to 180
df_sumup['longitude'] = (df_sumup.longitude + 180) % 360 - 180

# limiting precision
df_sumup['timestamp'] = df_sumup['timestamp'].dt.strftime('%Y-%m-%d').values
df_sumup.start_depth = df_sumup.start_depth.astype(float).round(4)
df_sumup.stop_depth = df_sumup.stop_depth.astype(float).round(4)
df_sumup.midpoint = df_sumup.midpoint.astype(float).round(4)
df_sumup.latitude = df_sumup.latitude.astype(float).round(6)
df_sumup.longitude = df_sumup.longitude.astype(float).round(6)

df_sumup.loc[df_sumup.elevation.isnull(), 'elevation'] = -9999
df_sumup['elevation'] = df_sumup.elevation.astype(int)
df_sumup['elevation'] = df_sumup['elevation'].astype(str).replace('-9999','')

for v in ['start_depth', 'stop_depth', 'midpoint', 'error']:
    df_sumup.loc[df_sumup[v]==-9999, v] = np.nan
    
df_sumup.density = df_sumup.density.astype(float).round(3)

write_variables = ['profile_key', 'reference_key', 'method_key', 'timestamp', 'latitude', 
  'longitude', 'elevation', 'start_depth', 'stop_depth', 'midpoint', 'density', 'error']

df_sumup.loc[df_sumup.latitude>0, write_variables].to_csv('SUMup 2023 beta/SUMup_2023_density_csv/SUMup_2023_density_greenland.csv', index=None)
df_sumup.loc[df_sumup.latitude<0, write_variables].to_csv('SUMup 2023 beta/SUMup_2023_density_csv/SUMup_2023_density_antarctica.csv', index=None)

import shutil
shutil.make_archive('SUMup 2023 beta/SUMup_2023_density_csv',
                    'zip', 'SUMup 2023 beta/SUMup_2023_density_csv')

# %% writing NetCDF files
df_sumup[['elevation']] =  df_sumup[['elevation']].replace('','-9999').astype(int)
df_sumup[df_sumup==-9999] = np.nan
import xarray as xr

def write_netcdf(df_sumup, filename):
    df_new = df_sumup.copy()
    df_new['timestamp'] = pd.to_datetime(df_new.timestamp).dt.tz_localize(None)

    df_new.index.name='measurement_id'
    ds_meta_name = (df_new[['profile_key','profile']]
                    .drop_duplicates()
                    .set_index('profile_key')
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
    
    ds_sumup = df_new[['profile_key', 'reference_key', 'method_key', 'timestamp',
           'latitude', 'longitude', 'elevation', 'start_depth',
           'stop_depth', 'midpoint', 'density', 'error']].to_xarray()
    
    ds_meta = xr.merge((
                        ds_meta_name,
                        ds_meta_method,
                        ds_reference_method))
    
    ds_sumup['elevation'] = ds_sumup.elevation.astype(int)
    ds_sumup['error'] = ds_sumup['error'].astype(float)  
    # ds_sumup['notes'] = ds_sumup['notes'].astype(str)   
    ds_sumup.timestamp.encoding['units'] = 'days since 1900-01-01'
    
    ds_meta['profile'] = ds_meta['profile'].astype(str)      
    ds_meta['method'] = ds_meta['method'].astype(str)      
    ds_meta['reference'] = ds_meta['reference'].astype(str)      
    ds_meta['reference_short'] = ds_meta['reference_short'].astype(str)      
    
    # adding attributes
    df_attr = pd.read_csv('doc/attributes_density.csv',
                          skipinitialspace=True,
                          comment='#').set_index('var')
    for v in df_attr.index:
        for c in df_attr.columns:
            if v in ds_sumup.keys():
                ds_sumup[v].attrs[c] = df_attr.loc[v,c]
            else:
                ds_meta[v].attrs[c] = df_attr.loc[v,c]
            
    if ds_sumup.latitude.isel(measurement_id=0)>0:
        ds_sumup.attrs['title'] = 'SUMup density dataset for the Greenland ice sheet (2023 release)'
    else:
        ds_sumup.attrs['title'] = 'SUMup density dataset for the Antarctica ice sheet (2023 release)'
    ds_sumup.attrs['contact'] = 'Baptiste Vandecrux'
    ds_sumup.attrs['email'] = 'bav@geus.dk'
    ds_sumup.attrs['production date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
       
    float_encoding = {"dtype": "float32", "zlib": True,"complevel": 9}
    int_encoding = {"dtype": "int32", "_FillValue":-9999, "zlib": True,"complevel": 9}
    
    ds_sumup.to_netcdf(filename,  
                       group='DATA',
                       encoding={
                          "measurement_id": int_encoding,
                          "density": float_encoding|{'least_significant_digit':2},
                          "start_depth": float_encoding|{'least_significant_digit':4},
                          "stop_depth": float_encoding|{'least_significant_digit':4},
                          "midpoint": float_encoding|{'least_significant_digit':4},
                          "error": float_encoding|{'least_significant_digit':4},
                          "longitude": float_encoding|{'least_significant_digit':4},
                          "latitude": float_encoding|{'least_significant_digit':4},
                          "elevation": int_encoding,
                          "profile_key": int_encoding,
                          "reference_key": int_encoding,
                          "method_key": int_encoding,
                          })
    ds_meta.to_netcdf(filename,  
                       group='METADATA',mode='a',
                       encoding={
                          "profile_key": int_encoding,
                          "profile": {"zlib": True,"complevel": 9},
                          "reference_key": int_encoding,
                          "reference": {"zlib": True,"complevel": 9},
                          "reference_short": {"zlib": True,"complevel": 9},
                          "method_key": int_encoding,
                          "method": {"zlib": True,"complevel": 9},
                          })
    
    
write_netcdf(df_sumup.loc[df_sumup.latitude>0, :], 'SUMup 2023 beta/SUMup_2023_density_greenland.nc')
write_netcdf(df_sumup.loc[df_sumup.latitude<0, :], 'SUMup 2023 beta/SUMup_2023_density_antarctica.nc')

#%% creating tables for ReadMe file
import matplotlib.pyplot as plt
df_sumup['timestamp'] = pd.to_datetime(df_sumup.timestamp,utc=True)
   
df_meta = pd.DataFrame()
df_meta['total'] = [df_sumup.shape[0]]
df_meta['added'] = df_sumup.shape[0]-len_sumup_2022
df_meta['nr_references'] = str(len(df_sumup.reference.unique()))
df_meta['greenland'] = df_sumup.loc[df_sumup.latitude>0].shape[0]
df_meta['antarctica'] = df_sumup.loc[df_sumup.latitude<0].shape[0]
df_meta.index.name='index'
df_meta.to_csv('doc/ReadMe_2023_src/tables/density_meta.csv')

print('{:,.0f}'.format(df_sumup.shape[0]).replace(',',' ') +\
      ' density observations in SUMup 2023')
print('{:,.0f}'.format(df_sumup.shape[0]-len_sumup_2022).replace(',',' ') +\
      ' more than in SUMup 2022')
print('from '+ str(len(df_sumup.reference_short.unique())) + ' sources')
print('representing '+ str(len(df_sumup.reference.unique()))+' references')

print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude>0].shape[0]).replace(',',' ')+' observations in Greenland')
print('{:,.0f}'.format(df_sumup.loc[df_sumup.latitude<0].shape[0]).replace(',',' ')+' observations in Antarctica')


plot_dataset_composition(df_sumup.loc[df_sumup.latitude>0], 
        'doc/ReadMe_2023_src/figures/density_dataset_composition_greenland.png')
plot_map(df_sumup.loc[df_sumup.latitude>0,['latitude','longitude']].drop_duplicates(),
         'doc/ReadMe_2023_src/figures/density_map_greenland.png', 
         area='greenland')

plot_dataset_composition(df_sumup.loc[df_sumup.latitude<0], 
        'doc/ReadMe_2023_src/figures/density_dataset_composition_antarctica.png')

plot_map(df_sumup.loc[df_sumup.latitude<0,['latitude','longitude']].drop_duplicates(),
         'doc/ReadMe_2023_src/figures/density_map_antarctica.png', 
         area='antarctica')

print_table_dataset_composition(df_sumup.loc[df_sumup.latitude>0]).to_csv('doc/ReadMe_2023_src/tables/composition_density_greenland.csv',index=None)

print_table_dataset_composition(df_sumup.loc[df_sumup.latitude<0]).to_csv('doc/ReadMe_2023_src/tables/composition_density_antarctica.csv',index=None)


# export position table
df_sumup.loc[df_sumup.latitude>0, ['profile','profile_key','latitude','longitude','reference_short']].drop_duplicates().to_csv('doc/GIS/SUMup_2023_density_location_greenland.csv', index=None)

