# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import os

# import cartopy
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

def print_location_file(df_sumup, path_out):
    if 'timestamp' in df_sumup.columns:
        v_time = ['timestamp', 'timestamp']
    else:
        v_time = ['start_year','end_year']
        
    if 'profile' in df_sumup.columns:
        v_names = ['profile_key','profile']
    else:
        v_names = ['name_key','name']
        
    tmp = df_sumup[
            v_names + ['latitude', 'longitude','reference_key', 
                       'method_key'] + list(dict.fromkeys(v_time))
            ].groupby(['latitude','longitude'])
    
    df_loc = pd.DataFrame()
    
    for v in  v_names + ['reference_key', 'method_key']:
        df_loc['list_of_'+v+'s'] = tmp[v].unique().apply(list)

    df_loc['timestamp_min'] = tmp[v_time[0]].min()
    df_loc['timestamp_max'] = tmp[v_time[1]].max()
    df_loc['num_measurements'] = tmp['method_key'].count()
    
    for v in df_loc.columns:
        df_loc[v] = (df_loc[v].astype(str)
                    .str.replace('[','')
                    .str.replace(']','')
                    .str.replace(' 00:00:00+00:00',''))
        if 'key' in v:
            df_loc[v] = (df_loc[v].astype(str)
                        .str.replace(', ',' / ')
                        .str.replace('  ',' '))
        else:
            df_loc[v] = (df_loc[v].astype(str)
                        .str.replace('\', \'',' / ')
                        .str.replace(',','')
                        .str.replace('\'','')
                        .str.replace('  ',' '))
    df_loc.to_csv(path_out)


def parse_short_reference(df_in, verbose=False):
    df_sumup = df_in.copy()
    abc = 'bcdefghijklmnopqrstuvwxyz'
    df_sumup['reference'] = df_sumup.reference_full.astype(str)
    all_refs = np.array([['reference_short', 'reference']])
    for ref in df_sumup.reference.unique():
        if (',' not in ref) and (' ' not in ref):
            print(ref, 'left as is')
            ref_short = ref
        else:
            year = re.findall(r'\d{4}', ref)
            if len(year) > 0:
                year = year[0]
            else:
                year = ''
            
            # first guess
            name = ref.lstrip().split(',')[0].split(' ')[0]
            
            # some exceptions
            if name == 'U.S.': name = 'US_Army'
            if name == 'SIMBA': name = 'SIMBA: Lewis'
            if name == 'Paul': name = 'Smeets'
            if name == 'K.': name = 'Wegener'            
            if name in ['van', 'Van']: name = ref.lstrip().split(',')[0].replace(' ',' ')
            ref_short = name + ' et al. ('+ year+')'
            if name == 'US': ref_short = 'US ITASE: Mayewski and Dixon ('+ year+')'
            if name == 'Satellite-Era': ref_short = 'SEAT11: Brucker and Koenig ('+ year+')'
            
        count = 0
        while ref_short in all_refs[:,0]:
            if count == 0:
                ref_short = ref_short[:-1] + 'b)'
                # tmp = all_refs[-1][0]
                # all_refs[-1] = tmp[:-1] + 'a)'
                count = count + 1
            else:
                ref_short = ref_short[:-2] + abc[count] +')'
                count = count + 1
        if verbose: print(ref_short)
        all_refs = np.vstack([all_refs, [ref_short, ref]])
    df_ref = pd.DataFrame(all_refs[1:,:], columns=all_refs[0,:])
    return df_ref


def check_conflicts(df_sumup, df_new, var=['name', 'depth','temperature'],
                    verbose=1):
    coords_sumup = df_sumup[['latitude','longitude']].drop_duplicates()
    coords_new = df_new[['latitude','longitude']].drop_duplicates()
    diff_lat = np.abs(coords_sumup.latitude.values - coords_new.latitude.values[:, np.newaxis])
    diff_lon = np.abs(coords_sumup.longitude.values - coords_new.longitude.values[:, np.newaxis])
    
    potential_duplicates = np.where( (diff_lat < 0.01) & (diff_lon < 0.01))
    for k in range(len(potential_duplicates[0])):
        i = potential_duplicates[0][k]
        j = potential_duplicates[1][k]
        tmp = pd.DataFrame()
        tmp['in SUMup']= df_sumup.loc[coords_sumup.iloc[j,:].name,:].T
        tmp['in new dataset'] = df_new.loc[coords_new.iloc[i,:].name,:].T
        if np.round(tmp.loc['date','in SUMup']/10000) == np.round(tmp.loc['date','in new dataset']/10000):
            if verbose:
                print('\nPotential duplicate found:\n')
                
                print(tmp.loc[var])
                print('reference in SUMup:')
                print(tmp.loc['reference_full', 'in SUMup'])
                print('reference in new dataset:')
                print(tmp.loc['reference_full', 'in new dataset'])
            return [tmp.loc[var[0], 'in SUMup']]
    return []


def print_table_dataset_composition(df_in):
    df_sumup = df_in.copy()
    df_sumup['coeff'] = 1
    if 'start_year' in df_sumup.columns:
        df_summary =(df_sumup.groupby('reference_short')
                          .apply(lambda x: x.start_year.min())
                          .reset_index(name='start_year'))
        df_summary['end_year'] =(df_sumup.groupby('reference_short')
                            .apply(lambda x: x.end_year.max())
                            .reset_index(name='end_year')).end_year
    else:
        df_summary =(df_sumup.groupby('reference_short')
                          .apply(lambda x: x.timestamp.dt.year.min())
                          .reset_index(name='start_year'))
        df_summary['end_year'] =(df_sumup.groupby('reference_short')
                            .apply(lambda x: x.timestamp.dt.year.max())
                            .reset_index(name='end_year')).end_year
    df_summary['num_measurements'] = (df_sumup.groupby('reference_short')
                              .apply(lambda x: x.coeff.sum())
                              .reset_index(name='num_measurements')).num_measurements
    # df_summary['reference_key'] = (df_sumup.groupby('reference_short')
    #                           .reference_key.unique().apply(list)
    #                           .astype(str).str.replace("[","").str.replace("]","")
    #                           .reset_index(name='reference_keys')).reference_keys
    
    return df_summary.sort_values('start_year')
    
    
def plot_dataset_composition(df_in, path_to_figure='figures/dataset_composition.png'):
    df_sumup = df_in.copy()
    df_sumup['coeff'] = 1
    df_summary = (df_sumup.groupby('reference_short')
                  .apply(lambda x: x.coeff.sum())
                  .reset_index(name='num_measurements'))
    df_summary=df_summary.sort_values('num_measurements')
    explode = 0.2 + 0.3*(df_summary.num_measurements.max() - df_summary.num_measurements)/df_summary.num_measurements.max()
    
    cmap = plt.get_cmap("tab20c")
    
    fig, ax=plt.subplots(1,1, figsize=(12,8))
    # print(df_summary.shape[0],  (1-np.exp(-df_summary.shape[0]/500)))
    plt.subplots_adjust(bottom= 0.05)
    patches, texts = plt.pie( df_summary.num_measurements,
                             startangle=90,
                             explode=explode,
                             colors=plt.cm.tab20.colors)
    labels = df_summary.reference_short.str.replace(';','\n') + ' ' + (df_summary.num_measurements/df_summary.num_measurements.sum()*100).round(3).astype(str) + ' %'
    sort_legend = True
    if sort_legend:
        patches, labels, dummy =  zip(*sorted(zip(patches, labels,
                                                  df_summary.num_measurements),
                                              key=lambda x: x[2],
                                              reverse=True))
    
    plt.legend(patches, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2),
               fontsize=15, ncol=2, title='Data origin (listed clock-wise)',
               columnspacing=0.5,title_fontsize='xx-large')
    plt.ylabel('')
    
    plt.savefig(path_to_figure,dpi=300, bbox_inches='tight')
    
# def plot_map(df, filename, area='greenland'):
#     fig = plt.figure(figsize=(5,5))
#     if area=='greenland':
#         ax = plt.subplot(1,1,1,projection=ccrs.NorthPolarStereo(central_longitude=-45))
#         ax.set_extent([-56.5, -30, 59, 84], ccrs.PlateCarree())
#         land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
#                                             edgecolor='none',facecolor='darkgray')
#         ax.add_feature(land,zorder=1)
#     else:
#         ax = plt.subplot(1,1,1,projection=ccrs.SouthPolarStereo())
#         ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())
#         iceshelves = cfeature.NaturalEarthFeature('physical','antarctic_ice_shelves_polys','50m',
#                                                   edgecolor='none',facecolor='whitesmoke')
#         ax.add_feature(iceshelves,zorder=1)
#     ax.axis('off')
    
#     # Add features
#     glacier = cfeature.NaturalEarthFeature('physical','glaciated_areas','50m',
#                                            edgecolor='none',facecolor='w')
#     ax.add_feature(glacier,zorder=0)

#     ax.coastlines(color='gray',zorder=2)
    
#     # Plot locations
#     ax.scatter(df.longitude,df.latitude,
#                transform=ccrs.PlateCarree(),zorder=3,
#                c='#0BD0D9',s=40,edgecolor='gray',lw=0.5)
    
#     plt.tight_layout()
#     fig.savefig(filename,dpi=500)
import geopandas as gpd

def plot_map(df, filename, area='greenland'):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

    fig = plt.figure(figsize=(5,5))
    if area=='greenland':
        land = gpd.read_file('doc/GIS/greenland_land_3413.shp')
        ice = gpd.read_file('doc/GIS/greenland_ice_3413.shp')
        crs = 3413
    else:
        ice = gpd.read_file('doc/GIS/Medium_resolution_vector_polygons_of_the_Antarctic_coastline.shp')
        crs = 3031 
        ice = ice.to_crs(crs)
        land = ice.loc[(ice.surface=='land') & (ice.area<5000)]
        ice = ice.loc[(ice.surface!='land') | (ice.area>5000)]
        
    gdf = gdf.to_crs(crs)
    
    land.plot(ax=plt.gca(),color='k')
    ice.plot(ax=plt.gca(), color='lightblue')
    gdf.plot(ax=plt.gca(), marker='d', markersize=15, alpha=0.5, edgecolor='tab:blue', facecolor='None')
    plt.gca().axis('off')   
    plt.tight_layout()
    fig.savefig(filename,dpi=200)


def readSnowEx(file_name, sheet_name):
    df = pd.read_excel(file_name, sheet_name=sheet_name)

    """
    indexing convention df.iloc[row, column]:
    rows: index = row number - 2
    columns: 0 = A, 1 = B, ...
    example cell A3 = df.iloc[1, 0]
    """
    
    # indexing metadata:
    site_id = df.iloc[1, 0]
    site_lat = df.iloc[6, 7]
    site_lon = df.iloc[6, 11]
    site_elev = df.iloc[6, 16]
    
    time = df.iloc[6, 0]
    observer = df.iloc[1, 7]
    gps_uncertainty = df.iloc[6, 18]
    comments = df.iloc[1, 20]

    pss = df.iloc[6, 4]
    profile_temp_start = df.iloc[4, 18]
    profile_temp_end = df.iloc[4, 19]
    
    # reformat date into year, month, and day to fit into SWE summary table:
    try: 
        date = df.iloc[4, 0]
        date = str(date)
        date = date.split(" ")[0]
        day = int(date.split("-")[2])
        month = int(date.split("-")[1])
        year = int(date.split("-")[0])
    except: # attempted to handle data without date information
        print("Incomplete date information in source data")
        day = np.nan
        month = np.nan
        year = np.nan
    
    # create metadata dataframes:
    observation_meta = pd.DataFrame(data=(site_id, site_lat, site_lon, site_elev,
                                          date, day, month, year, time, observer,
                                          gps_uncertainty, comments),
                                    index=["site_id", "site_lat", "site_lon", "site_elev",
                                           "date", "day", "month", "year", "time",
                                           "observer", "gps_uncertainty", 
                                           "comments"]).T
    profile_meta = pd.DataFrame(data=(pss, profile_temp_start, profile_temp_end),
                                index=["pss", "profile_temp_start", "profile_temp_end"]).T

    # indexing density and temperature data:
    depth_top = df.iloc[10:100, 0] # length of column extended to 100 to (hopefully) include the longest dataseries 
    depth_bottom = df.iloc[10:100, 2]
    density_a = df.iloc[10:100, 3]
    density_b = df.iloc[10:100, 4]
    density_note = df.iloc[10:100, 5]
    cutter = df.iloc[10:100, 6]
    temp_depth = df.iloc[10:100, 7]
    temp = df.iloc[10:100, 8]
    
    # create density dataframe:
    density = pd.DataFrame(
        data=(depth_top, depth_bottom, density_a, 
              density_b, density_note, cutter, temp_depth, temp),
        index=["depth_top", "depth_bottom", "density_a", "density_b",
               "density_note", "cutter", "temp_depth", "temp"]).T
    density.reset_index(inplace=True, drop=True)     
    
    df_snowex = pd.concat([observation_meta, profile_meta, density], axis=1)
    df_snowex.reset_index(inplace=True, drop=True) # reindex df_snowex
        
    return observation_meta, density


def add_Greenland_profiles(df_sumup, necessary_variables, df_ref, short_ref):
    # %% BRPC_2012
    # -Compiled in 2012 by the BYRD POLAR RESEARCH CENTER
    # Surface Elevation and Velocity Changes on the South Central Greenland Ice Sheet: 1980-2011 - Data Summary
    # Kenneth C. Jezek October 29, 2012
    # Byrd Polar Research Center The Ohio State University Columbus, Ohio
    print('## loading BPRC 1993 core')
    
    df_BPRC = pd.read_csv('data/density data/BRPC_2012/BPRC_TechReport_2012_01.csv', sep=';', skiprows=3, header=None).iloc[:,:-1].iloc[:,[0,1,-1]]
    
    df_BPRC.columns=['start_depth', 'thickness_cm','density_gcm2']
    df_BPRC['start_depth'] = df_BPRC['start_depth'] /100
    df_BPRC['stop_depth'] = df_BPRC.start_depth + df_BPRC.thickness_cm/100
    df_BPRC['midpoint'] = df_BPRC.start_depth + df_BPRC.thickness_cm/2/100
    df_BPRC[['start_depth']]
    # profile_name = 'station_2006_1993_core'
    df_BPRC['profile'] = df_sumup.profile.max() +1
    df_BPRC['density'] = 1000*df_BPRC.density_gcm2
    df_BPRC['reference'] = df_sumup.reference.max() +1
    df_BPRC['reference_full'] = 'Jezek, K.C. 2012. Surface Elevation and Velocity Changes on the South Central Greenland Ice Sheet: 1980-2011 - Data Summary. BPRC Technical Report No. 2012-01, Byrd Polar Research Center, The Ohio State University, Columbus, Ohio, 28 pages plus the data summary document.'
    df_BPRC['reference_short'] = 'BPRC 1993 in Jezek (2012)'
    df_BPRC['profile_name'] = 'BPRC_1993'
    df_BPRC['method'] = 4
    df_BPRC['method_str'] = 'ice or firn core section'
    df_BPRC['date'] = 19930101
    df_BPRC['timestamp'] = '1993-01-01'
    df_BPRC['latitude']= 65.29296
    df_BPRC['longitude']= 314.166266-360
    df_BPRC['elevation']= -9999
    df_BPRC['sdos_flag']= 0
    df_BPRC['error']= -9999
    sumup_index_conflict = check_conflicts(df_sumup, df_BPRC)
    
    df_sumup = pd.concat((df_sumup, df_BPRC[necessary_variables]), ignore_index=True)
    
    # %% EGIG historical
    print('## loading EGIG historical')
    xl = pd.ExcelFile('data/density data/EGIG historical/EGIG_bva.xlsx')
    ind_ref = df_sumup.reference.max() +1
    ind_profile =  df_sumup.profile.max() +1
    print('some data already in dataset')
    count = 0 
    for i, sheet in enumerate(xl.sheet_names):
        df_EGIG = pd.read_excel('data/density data/EGIG historical/EGIG_bva.xlsx',header=None, sheet_name=sheet)
        ref = df_EGIG.iloc[0,0]
        site = df_EGIG.iloc[0,1]
        if not isinstance(site, str):
            if np.isnan(site):
                site = 'Camp VI'
        if site in ['T53', 'T61', 'T43', 'T31', 'Camp VI']:
            print('skipping', site)
            if i == 0:
                # only updating profile name
                df_sumup.loc[df_sumup.profile==925, 'profile_name'] = 'Camp_VI_1959'
                df_sumup.loc[df_sumup.profile==926, 'profile_name'] = 'T31_1959'
                df_sumup.loc[df_sumup.profile==927, 'profile_name'] = 'T43_1959'
                df_sumup.loc[df_sumup.profile==928, 'profile_name'] = 'T53_1959'
                df_sumup.loc[df_sumup.profile==929, 'profile_name'] = 'T61_1959'
        else:
            print('adding',site)
            date = pd.to_datetime(df_EGIG.iloc[0,2]).strftime('%Y%m%d')
        
            df_EGIG = df_EGIG.iloc[2:,:4]
            df_EGIG.columns = ['start_depth', 'stop_depth','midpoint','density']
            df_EGIG[['start_depth', 'stop_depth','midpoint']] = df_EGIG[['start_depth', 'stop_depth','midpoint']]/100
            df_EGIG['profile'] = ind_profile + count
            count=count+1
            
            if ref == 'Renaud1969':
                df_EGIG['reference'] = 168
                df_EGIG['reference_full'] = df_ref.loc[168].values[0]
                df_EGIG['reference_short'] = short_ref.loc[df_ref.loc[168]].reference_short.values[0]
                df_EGIG['profile_name'] = site+' 1959'
    
            else:
                df_EGIG['reference'] = ind_ref
                df_EGIG['reference_full'] = 'Ambach 1970'
                df_EGIG['reference_short'] = 'Ambach 1970'
                df_EGIG['profile_name'] = 'Carrefour 1967'
            
            if df_EGIG.density.mean() < 1:
                df_EGIG.density = df_EGIG.density*1000
            df_EGIG['method'] = -9999
            df_EGIG['date'] = int(date)
            df_EGIG['timestamp'] = date[:4]+'-'+date[4:6]+'-'+date[6:8]
            df_EGIG[['latitude', 'longitude','elevation']] = np.nan
            
            # 69 49' 25" N., long. 47 25' 57"
            if site == 'T04': 
                lat, lon, elev = 69.82361, -47.4325, 1850
            if site == 'T15': 
                lat, lon, elev = 70.303, -44.57, 2491
            if site == 'T31': 
                lat, lon, elev = 70.909, -40.64, 3008
            if site == 'T43': 
                lat, lon, elev = 71.11666667, -37.31666667, 3174
            if site == 'T53': 
                lat, lon, elev = 71.35, -33.48333333, 2867
            if site == 'T61': 
                lat, lon, elev = 72.23333333, -32.33333333, 2750
            if site == 'Camp VI':
                lat, lon, elev = 69.7, -48.26666667, 1598
            df_EGIG['latitude'], df_EGIG['longitude'], df_EGIG['elevation'] = lat, lon, elev
        
            df_EGIG['sdos_flag']= 0
            df_EGIG['error']= -9999
            
            df_EGIG.set_index('start_depth').density.plot(label=site)
            plt.legend()
            plt.title('EGIG historical')
            sumup_index_conflict = check_conflicts(df_sumup, df_EGIG, 
                                                   var=['profile','date','start_depth', 'density'])
            df_sumup = pd.concat((df_sumup, df_EGIG[necessary_variables]), ignore_index=True)
    
    # %% NGT
    print('## loading NGT core')
    ind_ref
    df_NGT_meta = pd.read_csv('data/density data/NGT/metadata_NGT.csv', sep=';')
    dir_list = os.listdir('data/density data/NGT/DEN')[:-1]
    for i, core_name in enumerate(dir_list):
        if core_name in ['B16','B17','B18','B21','B26','B29']:
            print('skipping', core_name)
            # renaming giving names to the NGT profiles already in SUMup 2022
            if i == 0:
                for ref, name in zip([24, 25,26,27,28,29],
                                     ['ngt03C93.2_B16', 'ngt06C93.2_B17', 'ngt14C93.2_B18',
                                      'ngt27C94.2_B21','ngt37C95.2_B26','ngt42C95.2_B29']):
                    df_sumup.loc[df_sumup.reference==ref,'profile_name'] = name
                    print('Naming profile',
                          df_sumup.loc[df_sumup.reference==ref,'profile'].values[0],
                          name)
            continue
        try:
            df_NGT = pd.read_csv('data/density data/NGT/DEN/'+core_name+'/'+core_name+'-den.10_cm', sep='\t')#.iloc[:,:-2]
        except:
            df_NGT = pd.read_csv('data/density data/NGT/DEN/'+core_name+'/'+core_name+'-den', sep='\t').iloc[:,:-2]
        df_NGT = df_NGT.iloc[2:,:4]
        df_NGT.columns = ['start_depth', 'density']
        df_NGT['stop_depth'] = df_NGT.start_depth+0.001
        df_NGT['midpoint'] = df_NGT.start_depth+0.0005
        df_NGT['density'] = df_NGT.density.astype(float).values * 1000
    
        df_NGT['profile'] = 2009 + i
        df_NGT['profile_name'] = df_NGT_meta.loc[df_NGT_meta.Eiskern == core_name,'Position'].values[0].lower()\
            +'C'+ str(df_NGT_meta.loc[df_NGT_meta.Eiskern == core_name,'Year'].values[0])[-2:] \
                +'_'+core_name
        print(core_name, df_NGT_meta.loc[df_NGT_meta.Eiskern == core_name,'Position'].values[0].lower()\
            +'C'+ str(df_NGT_meta.loc[df_NGT_meta.Eiskern == core_name,'Year'].values[0])[-4:-2] \
                +'_'+core_name)
        df_NGT['reference'] = 200
        df_NGT['reference_full'] = 'Wilhelms, F.: Measuring the Conductivity and Density of Ice Cores, Ber. Polarforsch., 191 pp., 1996.'
        df_NGT['reference_short'] = 'NGT: Wilhelms (1996)'
            
        df_NGT['method'] = 5
        df_NGT['method_str'] = 'Neutron density probe/MADGE'
        
        df_NGT['date'] = df_NGT_meta.loc[df_NGT_meta.Eiskern == core_name,'Year'].values[0]*10000 + 101
        df_NGT['timestamp'] = str(df_NGT['date'].unique()[0])[0:4]+'-'+str(df_NGT['date'].unique()[0])[4:6]+'-'+str(df_NGT['date'].unique()[0])[6:8]
    
        df_NGT['latitude'] = df_NGT_meta.loc[df_NGT_meta.Eiskern == core_name, 'Lat'].values[0]
        df_NGT['longitude'] = df_NGT_meta.loc[df_NGT_meta.Eiskern == core_name, 'Lon'].values[0]
        df_NGT['elevation'] = df_NGT_meta.loc[df_NGT_meta.Eiskern == core_name, 'Elev'].values[0]
    
        df_NGT['sdos_flag']= 0
        df_NGT['error']= -9999
        
        df_NGT.set_index('start_depth').density.plot(label=core_name)
        plt.legend()
        plt.title('NGT')
        sumup_index_conflict = check_conflicts(df_sumup, df_NGT, var=['profile','date','start_depth', 'density'])
        df_sumup = pd.concat((df_sumup, df_NGT[necessary_variables]), ignore_index=True)
    
    # %% Van Der Veen 2001
    print('loading Van der veen')
    df_VanVeen = pd.read_csv('data/density data/van der Veen et al 2001/van der Veen et al 2001.csv', 
                             sep = ';',skiprows=3, encoding='ISO-8859-1', header=None)
    df_VanVeen.loc[322] = np.nan
    df_Van_meta = pd.read_csv('data/density data/van der Veen et al 2001/metadata.csv', sep=';')
    
    df_Van_meta['latitude'] = df_Van_meta.lat1 +df_Van_meta.lat2/60 + df_Van_meta.lat3/60/60
    df_Van_meta['longitude'] = df_Van_meta.lon1 +df_Van_meta.lon2/60 + df_Van_meta.lon3/60/60
    df_Van_meta['Name'] =df_Van_meta['Name'].str.strip()
    df_Van_meta = df_Van_meta.rename(columns={'elev': 'elevation'})
    
    count=0
    plt.figure()
    
    for j, column_name in enumerate(df_VanVeen.columns):
        if (df_VanVeen.iloc[1, j] == 'Depth'):
            count=count+1
    
            van_df = pd.DataFrame(columns=necessary_variables)
            profile_name = df_VanVeen.iloc[0, j].strip()
            print(profile_name)
            length = df_VanVeen[j].last_valid_index()
          
            van_df['start_depth'] = df_VanVeen.iloc[2:(length+1), j].values.astype(float)
            van_df['density'] =  df_VanVeen.iloc[2:(length+1), j+1].values.astype(float)
            van_df['stop_depth'] = df_VanVeen.iloc[3:(length+2), j].values.astype(float)
            van_df['midpoint'] = van_df['start_depth'] + (van_df['stop_depth'] - van_df['start_depth'])/2
            
            van_df['profile_name'] = profile_name
            van_df['date'] = df_Van_meta.loc[df_Van_meta.Name == profile_name, 'year'].values[0]*1000
            van_df['timestamp'] = '%0.0f-01-01'%df_Van_meta.loc[df_Van_meta.Name == profile_name, 'year'].values[0]
            
            
            van_df['latitude'] = df_Van_meta.loc[df_Van_meta.Name == profile_name, 'latitude'].values[0]
            van_df['longitude'] = df_Van_meta.loc[df_Van_meta.Name == profile_name, 'longitude'].values[0]
            van_df['elevation'] = df_Van_meta.loc[df_Van_meta.Name == profile_name, 'elevation'].values[0]
    
            van_df['error'] = -9999
            van_df['reference'] = 201
            van_df['reference_full'] = 'van der Veen, C.J.,E. Mosley‐Thompson, K. C. Jezek , I. M. Whillans & J. F.Bolzan (2001) Accumulation rates in South and Central Greenland, Polar Geography, 25:2, 79-162,DOI: 10.1080/10889370109377709'
            van_df['reference_short'] = 'van der Veen et al. (2001)'
            van_df['method'] = 4
            van_df['profile'] = df_sumup.profile.max()+1
            
            van_df.set_index('start_depth').density.plot(drawstyle='steps', label=profile_name)
            sumup_index_conflict = check_conflicts(df_sumup, van_df, var=['profile','date','start_depth', 'density'])
            df_sumup = pd.concat((df_sumup, van_df[necessary_variables]), ignore_index=True)
    plt.legend()
    
    # %% GEUS snow pit and firn core dataset
    list_of_paths = os.listdir("data/density data/GEUS snow pit and firn core dataset/")
    list_of_paths = [ "data/density data/GEUS snow pit and firn core dataset/"+f for f in list_of_paths]
    print('Loading GEUS snow pit and firn core dataset')
    # option to include a density vs. depth plot:
    do_plot = 0 # change to 1 to produce plot
    l1 = len(df_sumup)
    ref_num_geus_snowpits = df_sumup.reference.max()+1
    
    for fn in list_of_paths:
        xl = pd.ExcelFile(fn)    
        sheets = xl.sheet_names
        print(fn.split('/')[-1])
        sentence_list=[]
        if do_plot:
            fig, ax = plt.subplots(figsize=(7, 6))
        for i, sheet in enumerate(sheets):
            print('   ',sheet)
            observation_meta, df = readSnowEx(file_name=fn, sheet_name=sheet)
            if observation_meta.day.isnull().all():
                continue
            if observation_meta.site_lat.isnull().sum():
                print('missing coordinates, skipping')
                continue
            if sheet == 'JAR1 profile 02':
                # duplicated density, but added stratigarphy info
                continue
            df = df.rename(columns={'depth_top':'start_depth',
                                    'depth_bottom':'stop_depth'})
            df['start_depth'] = df.start_depth/100
            df['stop_depth'] = df.stop_depth/100
            df['density'] = df[['density_a', 'density_b']].mean(axis=1)
            df['error'] = df[['density_a', 'density_b']].std(axis=1)
            df['midpoint'] = df.start_depth + (df.stop_depth-df.start_depth)/2
            
            df['profile'] = df_sumup.profile.max() + 1
            df['profile_name'] = fn.split('_')[1].split('.xlsx')[0]
            if len(sheets)>1:
                df['profile_name'] = fn.split('_')[1].split('.xlsx')[0]+'_'+sheet
                
            df['reference'] = ref_num_geus_snowpits
            df['reference_short'] = 'GEUS snow and firn data (2023)'
            df['reference_full'] = 'Vandecrux, B.; Box, J.; Ahlstrøm, A.; Fausto, R.; Karlsson, N.; Rutishauser, A.; Citterio, M.; Larsen, S.; Heuer, J.; Solgaard, A.; Colgan, W.: GEUS snow and firn data in Greenland, https://doi.org/10.22008/FK2/9QEOWZ , GEUS Dataverse, 2023'            
            # if 'Nanok' in fn:
            #     df['reference'] = ref_num_nanok
            #     df['reference_short'] = 'Nanok 2022 ski expedition'
            #     df['reference_full'] = 'Nanok 2022 ski expdition, https://www.nanokexpedition.be/'
            # else:
            #     df['reference'] = ref_num_gcn_snowpits
            #     df['reference_short'] = 'GC-Net snowpits'
            #     df['reference_full'] = 'Steffen, K., Heilig, K., McGrath, D., Bayou, N., Steffen, S., Houtz, D., Naderpour, R. : Historical GC-Net snowpit data in Greenland (unpublished)'        
            df['method'] = 6
    
            if observation_meta.time.isnull().all():
                df['timestamp'] = observation_meta.date.iloc[0]
            else:
                df['timestamp'] = observation_meta.date.iloc[0]+'T'+observation_meta.time.astype(str).iloc[0].replace(' (UTC',':00').replace(')','')
            df['date'] = int(observation_meta.date.iloc[0].replace('-',''))
            df['latitude'] = observation_meta.site_lat.iloc[0]
            df['longitude'] = -abs(observation_meta.site_lon.iloc[0])
            df['elevation'] = observation_meta.site_elev.iloc[0]
    
            sumup_index_conflict = check_conflicts(df_sumup, df,
                                    var=['profile','profile_name','date',
                                          'start_depth', 'density'])
            if len(sumup_index_conflict)>0:
                continue
            df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)        
            if do_plot:
                plt.plot(df["midpoint"], df["density"],marker='o', label=sheet)
        if do_plot:
            plt.title(fn.split('/')[-1])
            plt.legend()
            plt.show()
            plt.xlabel('Depth (cm)')
            plt.ylabel('Density (kg m-3)')
    
    
    print(len(df_sumup)-l1, 'observations added')   
    
    # %% Main PARCA cores, only giving names to profiles, data is already in
    print('Main PARCA cores, only giving names to profiles, data is already in')
    # option to include a density vs. depth plot:
    do_plot = 0 # change to 1 to produce plot
    l1 = len(df_sumup)
    for k, fn in enumerate(['data/density data/PARCA/PARCA-1997-cores.xls',
               'data/density data/PARCA/PARCA-1998-cores_bapt.xls']):
        xl = pd.ExcelFile(fn)
        sheets = xl.sheet_names
    
        if do_plot:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(7, 6))
        for i, profile_name in enumerate(sheets):
            if (k==0) & (i == 0):
                continue
            if profile_name in ['STUNUB','STUNUC','SDOMEB','NASAEASTB', 'NDYE3C']:
                continue
            df_raw =  pd.read_excel(fn, sheet_name=profile_name)
    
            df = pd.DataFrame()
            df['midpoint'] = pd.to_numeric(df_raw.iloc[5:, 0], errors='coerce')
            df['density'] = pd.to_numeric(df_raw.iloc[5:, 1], errors='coerce')
            half_thick = (df.midpoint.values[1:] - df.midpoint.values[:-1])/2
            half_thick = np.append(half_thick, half_thick[-1])
            df['start_depth'] = df.midpoint - half_thick
            df['stop_depth'] = df.midpoint + half_thick
    
            df = df.loc[df.notnull().any(axis=1)]
            # Ellen Mosley-Thompson:
            # 1997 cores: I was not in the field for the drilling of these cores: 
            # Joe McConnell and Roger Bales would have the dates - my name is 
            # listed because we did the iostopes and dust analyses; I am confident 
            # these were drilled in the April - May window 
            if k == 0: date = '1997-05-01'
            # Ellen Mosley-Thompson:
            # Joe McConnell and Roger Bales were drilling these short cores and 
            # would have the dates - my name is listed because we did the iostopes
            # and dust analyses; I am confident these were drilled in the 
            # April - May window 
            if k == 1: date = '1998-05-01';
            
            # Ellen Mosley-Thompson:
            # Core A May 13 - 16 Core B drilled on 12-05-1998
            if '(Raven) Core A' in profile_name: date = '1998-05-13'
            if '(Raven) Core B' in profile_name: date = '1998-05-12'
            df['timestamp'] = date
            df['date'] = int(date.replace('-',''))
            df['latitude'] = float(df_raw.iloc[0,3].replace(' N','').replace('N',''))
            df['longitude'] =  -np.abs(float(df_raw.iloc[1,3].replace(' W','').replace('W','')))

            df['profile'] = df_sumup.profile.max() + 1
            df['profile_name'] = profile_name
            # print(i, profile_name,df[['latitude', 'longitude']].drop_duplicates().values)
            # check if already in SUMup
            sumup_index_conflict = check_conflicts(df_sumup, df,
                                    var=['profile','profile_name','date',
                                         'midpoint', 'density'],
                                    verbose=0)

            if 'Raven' in profile_name:
                df['error'] = -9999
                df['reference'] = 7
                df['elevation'] = 2119
                df['reference_full'] = 'Mosley-Thompson, E., J.R. McConnell, R.C. Bales, Z. Li, P-N. Lin, K. Steffen, L.G. Thompson, R. Edwards, and D. Bathke, 2001, Local to Regional-Scale Variability of Greenland Accumulation from PARCA cores. Journal of Geophysical Research (Atmospheres), 106 (D24), 33,839-33,851. doi: 10.1029/2001JD900067'
                df['reference_short'] = 'Mosley-Thompson et al. (2001)'
                df['method'] = 4
            
                print('adding',profile_name,'as profile', df_sumup.profile.max() + 1)
                df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
            else:
                if sumup_index_conflict[0] == 67: 
                    profile_name = 'NDYE3A&B'
                if len(sumup_index_conflict)>0:
                    print('profile', sumup_index_conflict, 'from', 
                          df_sumup.loc[df_sumup.profile==sumup_index_conflict[0],
                                 'timestamp'].iloc[0],
                          'updated to', 'PARCA_'+profile_name+'_'+str(int(df['date'].iloc[0]/10000)), date)
                    df_sumup.loc[df_sumup.profile==sumup_index_conflict[0],
                                 'profile_name'] = 'PARCA_'+profile_name+'_'+str(int(df['date'].iloc[0]/10000))
                    df_sumup.loc[df_sumup.profile==sumup_index_conflict[0],
                                 'timestamp'] = date
                    df_sumup.loc[df_sumup.profile==sumup_index_conflict[0],
                                 'date'] = int(date.replace('-',''))
                    if do_plot:
                        plt.figure()
                        plt.plot(df.midpoint,df.density, marker='d', linestyle='None',
                                 label='raw')
                        plt.plot(df_sumup.loc[df_sumup.profile==sumup_index_conflict[0]].midpoint,
                                 df_sumup.loc[df_sumup.profile==sumup_index_conflict[0]].density,
                                 marker='o', linestyle='None',label='sumup', alpha=0.8)
                        plt.legend()
                        plt.title(profile_name)
                        plt.show()
                    
    # %% Secondary PARCA cores
    print('Secondary PARCA cores')
    do_plot = 0
    
    df_meta = pd.read_csv('data/density data/PARCA/secondary cores/metadata.csv')
    l1 = len(df_sumup)
    count = 0
    for k, fn in enumerate(['data/density data/PARCA/secondary cores/TUNU_1996_Main_Core_Density.xlsx',
               'data/density data/PARCA/secondary cores/TUNU_Shallow_Cores_Density_for_Baptiste_Vandecrux.xls']):
        xl = pd.ExcelFile(fn)
        sheets = xl.sheet_names
    
        if do_plot:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(7, 6))
        for i, profile_name in enumerate(sheets):
            df_raw =  pd.read_excel(fn, sheet_name=profile_name, header=None)
    
            df = pd.DataFrame()
            df['start_depth'] = pd.to_numeric(df_raw.iloc[3:, 2], errors='coerce')
            df['stop_depth'] = df['start_depth']+pd.to_numeric(df_raw.iloc[3:, 3], errors='coerce')/100
            df['midpoint'] = pd.to_numeric(df_raw.iloc[3:, 9], errors='coerce')
            df['density'] = pd.to_numeric(df_raw.iloc[3:, 8], errors='coerce')
            df = df.loc[df.notnull().any(axis=1)].iloc[:-1,:]
            print('   ', profile_name, df_meta.iloc[10+count,0])
            # Ellen Mosley-Thompson:
            # definitely in April -May
            df['timestamp'] = '%i-05-01'%df_meta.iloc[10+count,1]
            df['date'] = df_meta.iloc[10+count,1]*10000 + 501
            df['latitude'] = df_meta.iloc[10+count,2]
            df['longitude'] =  -abs(df_meta.iloc[10+count,3])
            df['elevation'] =  df_meta.iloc[10+count,4]
            df['profile'] = df_sumup.profile.max() + 1
            df['profile_name'] = df_meta.iloc[10+count,0]
    
            df['error'] = -9999
            df['reference'] = 7
            df['reference_full'] = 'Mosley-Thompson, E., J.R. McConnell, R.C. Bales, Z. Li, P-N. Lin, K. Steffen, L.G. Thompson, R. Edwards, and D. Bathke, 2001, Local to Regional-Scale Variability of Greenland Accumulation from PARCA cores. Journal of Geophysical Research (Atmospheres), 106 (D24), 33,839-33,851. doi: 10.1029/2001JD900067'
            df['reference_short'] = 'Mosley-Thompson et al. (2001)'
            df['method'] = 4
            
            count=count+1
            
            # check if already in SUMup
            sumup_index_conflict = check_conflicts(df_sumup, df,
                                    var=['profile','profile_name','date',
                                         'midpoint', 'density'],
                                    verbose=1)
            df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    
            if do_plot:
                plt.plot(df.midpoint,df.density, marker='d', linestyle='None',
                         label='raw')
                plt.legend()
                plt.title(profile_name)
                plt.show()
    
    count = 0    
    for k, fn in enumerate(['data/density data/PARCA/secondary cores/GITS_1996_Density_Data_for_Baptiste_Vandecrux.xlsx',
               'data/density data/PARCA/secondary cores/Humboldt_1995_cores_measured_densities_provided_to_Baptiste_Vandecrux.xls',
                          'data/density data/PARCA/secondary cores/NASA_U_Density_Data_for_Baptiste_Vandecrux.xls']):
        df_raw =  pd.read_excel(fn, header=None)
        if k == 0:
            ind = [0, 4, 8]
        elif k == 1:
            ind = [0, 3, 6, 9, 12, 15]
        elif k == 2:
            ind = [4]
            
        for j in ind:
            df = pd.DataFrame()
            df['start_depth'] = pd.to_numeric(df_raw.iloc[3:, j], errors='coerce')
            df['density'] = pd.to_numeric(df_raw.iloc[3:, j+1], errors='coerce')
            if k == 0:
                df['stop_depth'] = df['start_depth']+0.05
            else:
                df['stop_depth'] = np.nan
                df['stop_depth'].iloc[:-1] = df['start_depth'].iloc[1:]
            if k == 2:
                df['density'] = pd.to_numeric(df_raw.iloc[3:, 8], errors='coerce')
                df['density']=df['density']*1000
    
            df['midpoint']  = df['start_depth'] + (df['stop_depth'] - df['start_depth'])/2
            if (k==0) & (j != 8):
                df['midpoint'] = pd.to_numeric(df_raw.iloc[3:, j+1], errors='coerce')
                df['stop_depth'] = df['start_depth']+ (df['midpoint'] - df['start_depth'])*2
                df['density'] = pd.to_numeric(df_raw.iloc[3:, j+2], errors='coerce')
            df = df.loc[df.notnull().any(axis=1)].iloc[:-1,:]
            print('   ', df_meta.iloc[count,0])
            if k == 0:
                # E. Mosley-Thompson: GITS 1996 core drilled May 7 - 11
                assert df_meta.iloc[count,1] == 1996
                df['timestamp'] = '1996-05-10'
                df['date'] = 19960510
            if k == 1:
                # E. Mosley-Thompson: Humboldt 1995 drilled in April-May windows
                assert df_meta.iloc[count,1] == 1995
                df['timestamp'] = '1995-05-01'
                df['date'] = 19950501
            if k == 2:
                # E. Mosley-Thompson: NASA-U 1995 drilled in May 20 - 24
                assert df_meta.iloc[count,1] == 1995
                df['timestamp'] = '1995-05-20'
                df['date'] = 19950520
            df['latitude'] = df_meta.iloc[count,2]
            df['longitude'] =  -df_meta.iloc[count,3]
            df['elevation'] =  df_meta.iloc[count,4]
            df['profile'] = df_sumup.profile.max() + 1
            df['profile_name'] = df_meta.iloc[count,0]
    
            df['error'] = -9999
            df['reference'] = 7
            df['reference_full'] = 'Mosley-Thompson, E., J.R. McConnell, R.C. Bales, Z. Li, P-N. Lin, K. Steffen, L.G. Thompson, R. Edwards, and D. Bathke, 2001, Local to Regional-Scale Variability of Greenland Accumulation from PARCA cores. Journal of Geophysical Research (Atmospheres), 106 (D24), 33,839-33,851. doi: 10.1029/2001JD900067'
            df['reference_short'] = 'Mosley-Thompson et al. (2001)'
            df['method'] = 4
            
            count=count+1
        
            # check if already in SUMup
            sumup_index_conflict = check_conflicts(df_sumup, df,
                                    var=['profile','profile_name','date',
                                         'midpoint', 'density'],
                                    verbose=1)
            df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    
            if do_plot:
                plt.plot(df.midpoint,df.density, marker='d', linestyle='None',
                         label='raw')
                plt.legend()
                plt.title(df['profile_name'].drop_duplicates().iloc[0])
                plt.show()
    
    # %% Porter and Higgins core
    # Variables needed
    
    df = pd.read_csv('data/density data/PorterHiggins/Higgins_CP.csv', sep=';')
    
    df = df.rename(columns={'Depth at top of year in core (cm)': 'start_depth',
                            'Average Density (kg/m3)': 'density'}
                   )
    df['start_depth'] = df['start_depth']/100
    df['stop_depth'] = df['start_depth'] + df['thickness of the annual layer (cm)']/100
    df['midpoint']  = df['start_depth'] + (df['stop_depth'] - df['start_depth'])/2
    
    df['timestamp'] = '2007-01-01'
    df['date'] = 20070000
    df['latitude'] = 69.90
    df['longitude'] =  -47.00
    df['elevation'] =  2000
    df['profile'] = df_sumup.profile.max() + 1
    df['profile_name'] = 'CP_2007'
    df['reference'] = df_sumup.reference.max() + 1
    df['reference_full'] = 'Porter, S. & Mosley-Thompson, E., 2014. Exploring seasonal accumulation bias in a west central Greenland ice core with observed and reanalyzed data. Journal of Glaciology, 60(224), pp. 1065-1074.'
    df['reference_short'] = 'Porter and Mosley-Thompson (2014)'
    df['error'] = -9999
    df['method'] = 4
    sumup_index_conflict = check_conflicts(df_sumup, df,
                                            var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
    df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    
    
    # %% Morris and Wingham
    l = df_sumup.shape[0]
    path = 'data/density data/Morris and Wingham/'
    list_files = os.listdir(path);
    list_files = [f for f in list_files if f.endswith('xls')]
    ref_num = df_sumup.reference.astype(int).max() + 1
    
    df_sites=pd.read_excel(path+'metadata.xlsx', sheet_name='sites',header=None)
    df_sites.columns = ['site','latitude','longitude','elevation']
    df_sites = df_sites.set_index('site')
    
    df_files=pd.read_excel(path+'metadata.xlsx', sheet_name='files',header=None)
    df_files.columns=['file','site']
    df_files = df_files.set_index('file')
    
    k=0
    for f in list_files:
        df_raw = pd.read_excel(path+f, header=None)
        df = df_raw.iloc[65:,[11,15]]
        df.columns=['midpoint', 'density']
        df.midpoint = -pd.to_numeric(df.midpoint, errors='coerce')
        df['start_depth'] = df['midpoint']  - 0.005
        df['stop_depth'] = df['midpoint']  + 0.005
        
        df.density = pd.to_numeric(df.density, errors='coerce')*1000
        df = df.loc[df.midpoint>0, :]
        date=pd.to_datetime('%i-%s-%i'%tuple(df_raw.iloc[41:42,[5, 4, 6]].values[0].tolist()))
        df['timestamp'] = date.strftime('%Y-%m-%d')
        df['date'] = df['timestamp'].str.replace('-','').astype(int)
    
        df['profile'] = df_sumup.profile.max() + 1
    
        abc='abcdefg'
        profile_name = df_files.loc[f].values[0]+'_'+date.strftime("%Y_%b")
        i=-1
        while profile_name in df_sumup.profile_name.unique():
            i=i+1
            profile_name = df_files.loc[f].values[0]+'_'\
                +date.strftime("%Y_%b")+'_'+abc[i]
            
        df['profile_name'] = profile_name
        
        df['latitude'] = df_sites.loc[df_files.loc[f], 'latitude'].values[0]
        df['longitude'] =  -abs(df_sites.loc[df_files.loc[f], 'longitude']).values[0]
        df['elevation'] = df_sites.loc[df_files.loc[f], 'elevation'].values[0]
        
        df['error'] = -9999
        df['reference'] = ref_num
        df['reference_full'] = 'Morris, E.M. and D J Wingham. (2014) Densification of polar snow: measurements, modelling and implications for altimetry. JGR Earth Surfaces, doi: 10.1002/2013JF002898'
        df['reference_short'] = 'Morris and Wingham (2014)'
        df['method'] = 5
        
        # print(f, date.strftime('%Y-%m-%d'), profile_name)
        # k=k+1
        # if k == 5:
        #     plt.figure()
        #     k=0
        # plt.plot(df.density, -df.midpoint, marker='.', ls='None', label=profile_name)
        # plt.grid(); plt.xlabel('density'); plt.ylabel('depth');  plt.legend()
            
        # sumup_index_conflict = check_conflicts(df_sumup, df,
        #                                         var=['profile', 'date', 'start_depth', 'density'])
        df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    print('added', df_sumup.shape[0]-l,'measurements')
    
    # %% EGIG Fischer
    print('loading EGIG Fischer')
    meta = pd.read_excel('data/density data/Morris and Wingham/metadata.xlsx',header=None)
    meta.columns = ['site','latitude','longitude','elevation']
    meta = meta.set_index('site')
    
    file_list = [l for l in os.listdir('data/density data/EGIG_Fischer') if l.endswith('.DI')]
    ref_num = df_sumup.reference.astype(int).max() + 1
    for file in file_list:
        print('   ', file[:3])
        df = pd.read_csv('data/density data/EGIG_Fischer/'+file,
                         delim_whitespace=True, header=None)
        df.columns = ['start_depth', 'density']
        df.start_depth = pd.to_numeric(df.start_depth, errors='coerce')
        df = df.dropna()
    
        df.start_depth  = df.start_depth /100
        df['stop_depth'] = np.nan
        df.stop_depth[:-1] = df.start_depth.iloc[1:]
        df['midpoint'] = df.start_depth + (df.stop_depth-df.start_depth)/2
        df.density = df.density*1000
        df['timestamp'] = pd.to_datetime('1990-01-01')
        df['date'] = 19900101
        df['reference_full'] = 'Fischer, H., Wagenbach, D., Laternser, M. & Haeberli, W., 1995. Glacio-meteorological and isotopic studies along the EGIG line, central Greenland. Journal of Glaciology, 41(139), pp. 515-527.'
        df['reference_short'] = 'Fischer et al. (1995)'
        df['reference'] = ref_num
        df['profile_name'] = file[:3]
        df['profile'] = df_sumup.profile.max() + 1
        df['latitude'] = meta.loc[file[:3]].latitude
        df['longitude'] = -np.abs(meta.loc[file[:3]].longitude)
        df['elevation'] = meta.loc[file[:3]].elevation
        df['method'] = 4
        df['error'] = -9999
        
        # plt.figure()
        # plt.plot(df.density, -df.midpoint,'o',ls='None')
        # plt.title(file[:3])
        # plt.xlabel('density')
        # plt.ylabel('depth')
        
        # sumup_index_conflict = check_conflicts(df_sumup, df,
        #                                         var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
        df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    
    
    # %% Kameda core
    print('loading Kameda core')
    df = pd.read_excel('data/density data/Kameda 1991/Water eq depth of SiteJ ice core.xls')
    df = df.iloc[6:, [1, 2, 6]]
    
    df.columns = ['start_depth', 'stop_depth', 'density']
    df['reference_full'] = 'Kameda, T. et al. Melt features in ice cores from Site J, southern Greenland: some implications for summer climate since AD 1550. Ann. Glaciol. 21, 5158 (1995).'
    df['reference_short'] = 'Kameda et al. (1995)' 
    df['reference'] = df_sumup.reference.max() + 1
    df['profile'] = df_sumup.profile.max() + 1
    df['profile_name'] = 'Site J'
    df['latitude'] = 66.87
    df['longitude'] = -46.26
    df['elevation'] = 2030
    df['method'] = 4
    df['date'] = 19890401
    df['timestamp'] = pd.to_datetime('1989-06-01')
    df['midpoint'] = df.start_depth + (df.stop_depth-df.start_depth)/2
    df['error'] = -9999
    
    sumup_index_conflict = check_conflicts(df_sumup, df,
                                            var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
    df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    
    # %% Wegener
    df = pd.read_csv('data/density data/Wegener/snow_firn_density_Eismitte.csv', comment='#', sep=';')
    df.columns = ['Nr', 'timestamp', 'start_depth', 'stop_depth', 'comments', 'density']
    df['start_depth'] = df['start_depth']/100
    df['stop_depth'] = df['stop_depth']/100
    df['density'] = df['density']*1000
    df['reference_full'] = 'Sorge, E. Glaziologische Untersuchungen in Eismitte, 5. Beitrag. p62-263 in K. Wegener: Wissenschaftliche Ergebnisse der deutschen Grönland-Expedition Alfred Wegener 1929 und 1930/1931 Bd. III Glaziologie.  as in  Abermann, J., Vandecrux, B., Scher, S. et al. Learning from Alfred Wegener’s pioneering field observations in West Greenland after a century of climate change. Sci Rep 13, 7583 (2023). https://doi.org/10.1038/s41598-023-33225-9'
    df['reference_short'] = 'Sorge et al. (1935) as in Abermann et al. (2023)' 
    df['reference'] = df_sumup.reference.max() + 1
    df['latitude'] = 71.183
    df['longitude'] = -39.9333
    df['elevation'] = 3010
    df['method'] = -9999
    df['midpoint'] = df.start_depth + (df.stop_depth-df.start_depth)/2
    df['error'] = -9999
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.strftime('%Y%m%d').astype(str)
    df['profile'] = df_sumup.profile.max() + 1
    df['profile_name'] = 'Eismitte_'+df['timestamp'].dt.strftime('%Y-%m-%d')
    ind = df_sumup.profile.max() + 1
    for d in np.unique(df.timestamp):
        df.loc[df.timestamp == d, 'profile'] = ind
        ind = ind + 1
    
    sumup_index_conflict = check_conflicts(df_sumup, df,
                                            var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
    df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

    df = pd.read_csv('data/density data/Wegener/snow_density_transect.csv', comment='#', sep=';')
    meta = pd.read_csv('data/density data/Wegener/RA_alt_lat_lon.txt', sep='\t')
    meta.longitude = meta.set_index('RA').longitude.interpolate(method='slinear', fill_value="extrapolate").values
    meta.latitude = meta.set_index('RA').latitude.interpolate(method='slinear', fill_value="extrapolate").values
    
    df.columns = ['timestamp', 'RA','elevation', 'start_depth', 'stop_depth', 'density','pitno']
    df['start_depth'] = df['start_depth']/100
    df['stop_depth'] = df['stop_depth']/100
    df['density'] = df['density']
    df['reference_full'] = 'Jülg, H. Dichtebestimmungen und Schneesondierung auf der Route 1–400 km. Deutsche Grönland-Expedition Alfred Wegener 1929 und 1930/31. Wissenschaftliche Ergebnisse vol. 4.2, 329–345 (1939) as in Abermann, J., Vandecrux, B., Scher, S. et al. Learning from Alfred Wegener’s pioneering field observations in West Greenland after a century of climate change. Sci Rep 13, 7583 (2023). https://doi.org/10.1038/s41598-023-33225-9'
    df['reference_short'] = 'Jülg (1939) as in Abermann et al. (2023)' 
    df['reference'] = df_sumup.reference.max() + 1
    df['latitude'] = meta.set_index('RA').loc[df.RA.values, 'latitude'].values
    df['longitude'] = meta.set_index('RA').loc[df.RA.values, 'longitude'].values
    
    df['method'] = -9999
    df['midpoint'] = df.start_depth + (df.stop_depth-df.start_depth)/2
    df['error'] = -9999
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.strftime('%Y%m%d').astype(str)
    df['profile_name'] = 'Wegener_RA'+df.RA.astype(str)+'_'+df['timestamp'].dt.strftime('%Y-%m-%d')
    ind = df_sumup.profile.max() + 1
    tmp = df[['RA','timestamp']].drop_duplicates()
    for RA, ts in zip(tmp.RA, tmp.timestamp):
        df.loc[(df.RA == RA) & (df.timestamp == ts), 'profile'] = ind
        ind = ind + 1
    sumup_index_conflict = check_conflicts(df_sumup, df,
                                            var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
    df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    
    # %% NEEM and NGRIP cores
    meta = pd.read_csv('data/density data/NEEM and NGRIP/metadata.txt',header=None, sep='|', skipinitialspace=True).T
    plt.figure()
    
    for dat in zip(meta.values):
        f, year, name, ref_short, ref, lat, lon, elev = tuple(dat[0].tolist())
        
        print('loading',name,'core')
        
        col = ['midpoint', 'density']
        skip = 0
        sep = '\t'
    
        if name == '2009S2_Keegan': 
            sep = ','
            col = ['midpoint', 'density', 'permeability']
        if name == 'NG2012': col = ['density', 'midpoint']; skip=18
        if sep != ' ': delim_whitespace=False
        df = pd.read_csv('data/density data/NEEM and NGRIP/'+f, 
                         sep = sep,
                         skiprows=skip)
        df.columns = col
        
        if df.density.median() < 10:
            df['density'] = df.density*1000
        df['reference_full'] = ref
        df['reference_short'] = ref_short
        df['reference'] = df_sumup.reference.max() + 1
        df['profile'] = df_sumup.profile.max() + 1
        df['profile_name'] = name
        df['latitude'] = pd.to_numeric(lat)
        df['longitude'] = pd.to_numeric(lon)
        df['elevation'] = pd.to_numeric(elev)
        df['method'] = 4
        year = pd.to_numeric(year)
        df['date'] = year*10000
        df['timestamp'] = pd.to_datetime(str(year)+'-06-01')
        half_thick = (df.midpoint.values[1:] - df.midpoint.values[:-1])/2
        half_thick = np.append(half_thick, half_thick[-1])
        df['start_depth'] = df.midpoint - half_thick
        df['stop_depth'] = df.midpoint + half_thick
        df['error'] = -9999
        
        plt.plot(df.density, -df.midpoint, label=name)
        plt.legend()
        
        sumup_index_conflict = check_conflicts(df_sumup, df,
                                                var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
        df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    
    # renaming other NEEM core in the dataset
    df_sumup.loc[df_sumup.reference == 11, 'profile_name']  = 'NEEM2009S2_Baker'
    df_sumup.loc[df_sumup.reference == 11, 'longitude'] = -51.06 # some rounding error in the last sumup release
    
    # %% Miège cores
    #  Remove weird data listed as Miège et al.
    df_sumup = df_sumup.loc[df_sumup.reference != 5]
    meta = pd.read_csv('data/density data/Miège/metadata.csv').set_index('Name')
    meta.Year = meta.Year.astype(int)
    xl = pd.ExcelFile('data/density data/Miège/ACT_FA_densities.xls')
    ref_num = df_sumup.reference.astype(int).max() + 1
    count = 0 
    for i, sheet in enumerate(xl.sheet_names):
        print('  ',sheet)
        if sheet == 'FA13B':
            print('already in SUMup, only giving profile name')
            df_sumup.loc[df_sumup.profile==57,'profile_name'] = 'FA13B'
            continue
        df = pd.read_excel('data/density data/Miège/ACT_FA_densities.xls', sheet)
        df.columns = ['stop_depth', 'density']
        df['start_depth'] = 0
        df.iloc[1:,df.columns.get_loc('start_depth')] = df.stop_depth[:-1]
        df.density = df.density*1000
        df['reference_full'] = 'Miège, C., R. R. Forster, J. E. Box, E. W. Burgess, J. R. McConnell, D. R. Pasteris, and V. B. Spikes (2013), 2010 Arctic Circle Traverse - Southeast Greenland high accumulation rates derived from firn cores and ground-penetrating radar, Annals of Glaciology, 54(63), 322–332, doi:10.3189/2013AoG63A358.'
        df['reference_short'] = 'Miège et al. (2013)' 
        df['reference'] = 5
        df['profile'] = df_sumup.profile.max() + 1
        df['profile_name'] = sheet
        df['latitude'] = meta.loc[sheet].Y
        df['longitude'] = meta.loc[sheet].X
        df['elevation'] = -9999
        df['method'] = 4
        df['date'] = meta.loc[sheet].Year*10000+101
        df['timestamp'] = pd.to_datetime('%i-01-01'%meta.loc[sheet].Year)
        df['midpoint'] = df.start_depth + (df.stop_depth-df.start_depth)/2
        df['error'] = -9999
        sumup_index_conflict = check_conflicts(df_sumup, df,
                                                var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
        df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    
        # if len(sumup_index_conflict)>0:
        # for i in range(57,63):
        #     plt.figure()
        #     # plt.plot(df.density, -df.midpoint,'o',ls='None', label='new data')
        #     tmp = df_sumup.loc[df_sumup.profile == i, :]
        #     print(i, tmp[['reference','latitude','longitude']].drop_duplicates().values)
        #     plt.plot(tmp.density, -tmp.midpoint,'d',ls='None', label='already in sumup')
        #     plt.title(i)
        #     plt.xlabel('density')
        #     plt.ylabel('depth')    
        #     plt.legend()
        #     print(
    
    # %% Spencer cores 
    df = pd.read_excel('data/density data/Spencer/firn depth_density data with mean accum_temp and elev.xls')
    df.columns = ['site_num', 'start_depth', 'density','acc','elevation','temp']
    df = df.set_index('site_num')
    meta = pd.read_excel('data/density data/Spencer/site information.xlsx')
    meta = meta.set_index('profile_num')
    meta['date'] = meta.date.astype(str)
    
    meta.lat=meta.lat.str.strip()+'0’ 0”'
    degrees = meta.lat.str.split('°').str[0].astype(float)
    minutes = meta.lat.str.split('°').str[1].str.split('’').str[0].astype(float)
    seconds = meta.lat.str.split('°').str[1].str.split('’').str[1].str.split('”').str[0].astype(float)
    meta['latitude'] = (degrees + minutes/60 +seconds/60/60) * ((meta['NS']=='N')*2-1)
        
    meta.lon=meta.lon.str.strip()+'0’ 0”'
    degrees = meta.lon.str.split('°').str[0].astype(float)
    minutes = meta.lon.str.split('°').str[1].str.split('’').str[0].astype(float)
    seconds = meta.lon.str.split('°').str[1].str.split('’').str[1].str.split('”').str[0].astype(float)
    meta['longitude'] = (degrees + minutes/60 +seconds/60/60) * ((meta['EW']=='E')*2-1) 
    meta.loc[meta['longitude'] > 180,'longitude'] = meta.loc[meta['longitude'] > 180,'longitude']-360
    del degrees, minutes, seconds
    
    ref_start = df_sumup.reference.max() + 1
    profile_start = df_sumup.profile.max() + 1
    
    df['stop_depth'] = np.append(df.start_depth.iloc[1:].values, np.nan)
    df['midpoint'] = df.start_depth + (df.stop_depth-df.start_depth)/2
    df['reference_short'] = np.nan

    for ind in meta.index:
        if (df.index == ind).sum() < 2:
            continue
        df.loc[ind,'latitude'] = meta.loc[ind,'latitude']
        if np.isnan(meta.loc[ind,'latitude']): df.loc[ind,'latitude'] = meta.loc[ind,'lat2'] -360
        df.loc[ind,'longitude'] = meta.loc[ind,'longitude']
        if np.isnan(meta.loc[ind,'latitude']): df.loc[ind,'longitude'] = meta.loc[ind,'lon2'] -360
        
        tmp = (meta.loc[ind,'reference_short'] + ' as in Spencer et al. (2001)' == df.reference_short)
        if tmp.any():
            df.loc[ind,'reference'] = df.loc[tmp, 'reference'].iloc[0]
        else:
            df.loc[ind,'reference'] = ref_start
            ref_start = ref_start+1
        df.loc[ind,'profile'] = profile_start
        profile_start = profile_start+1
        df.loc[ind,'reference_full'] = meta.loc[ind,'reference_full'] + ' as in Spencer, M. K., Alley, R. B., and Creyts, T. T.: Preliminary firn-densification model with 38-site dataset, J. Glaciol., 47, 671–676, https://doi.org/10.3189/172756501781831765, 2001.'
        df.loc[ind,'reference_short'] = meta.loc[ind,'reference_short'] + ' as in Spencer et al. (2001)'
        df.loc[ind,'profile_name'] = meta.loc[ind,'profile_name']
        print(df.loc[ind,'reference'].unique() , meta.loc[ind,'reference_short'] + ' as in Spencer et al. (2001)')
        if meta.loc[ind,'date'] == 'NaT':
            if isinstance(meta.loc[ind,'ref_org'],float):
                print('>>> No year info, no reference', 
                      ' '.join(meta.loc[ind,['profile_name','ref_org','reference_short']].astype(str).values.tolist()))
                continue
            else:
                print('>>> No year info, using reference', 
                      ' '.join(meta.loc[ind,['profile_name','ref_org','reference_short']].astype(str).values.tolist()))
                meta.loc[ind,'date'] = re.findall(r'\d+', 
                  meta.loc[ind,'ref_org'] + meta.loc[ind,'reference_short'])[0]+'-01-01'
        df.loc[ind,'date'] = int(meta.loc[ind,'date'].replace('-',''))
        df.loc[ind,'timestamp'] = pd.to_datetime(meta.loc[ind,'date'])  
        # plt.figure()
        # plt.plot(df.loc[ind,'density'],-df.loc[ind,'start_depth'], marker='o')
        # plt.title(df.loc[ind,'profile_name'].iloc[0])
        
        
    df['error'] = np.nan
    df['method'] = -9999
    df['method_str'] = np.nan
    sumup_index_conflict = check_conflicts(df_sumup, df,
                                            var=['profile', 'profile_name', 
                                                  'date', 'start_depth', 'density'])
    df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
 
    # %% Vallelonga cores at NEGIS (EastGRIP)
    print('loading Vallelonga cores at NEGIS (EastGRIP) and NGRIP')
    ind_ref  = df_sumup.reference.max() + 1
    df = pd.read_excel('data/density data/Vallelonga/NEGIS-Density-accumulation.xlsx')
    df = df.iloc[6:, 0:2]
    
    df.columns = ['midpoint', 'density']
    df['start_depth'] = df.midpoint - 0.55/2
    df['stop_depth'] = df.midpoint + 0.55/2
    df['density'] = df.density *1000
    
    df['reference_full'] = 'Vallelonga, P., Christianson, K., Alley, R. B., Anandakrishnan, S., Christian, J. E. M., Dahl-Jensen, D., Gkinis, V., Holme, C., Jacobel, R. W., Karlsson, N. B., Keisling, B. A., Kipfstuhl, S., Kjær, H. A., Kristensen, M. E. L., Muto, A., Peters, L. E., Popp, T., Riverman, K. L., Svensson, A. M., Tibuleac, C., Vinther, B. M., Weng, Y., and Winstrup, M.: Initial results from geophysical surveys and shallow coring of the Northeast Greenland Ice Stream (NEGIS), The Cryosphere, 8, 1275–1287, https://doi.org/10.5194/tc-8-1275-2014, 2014.'
    df['reference_short'] = 'Vallelonga et al. (2014)' 
    df['reference'] = ind_ref
    df['profile'] = df_sumup.profile.max() + 1
    df['profile_name'] = 'EastGRIP'
    df['latitude'] = 75.623
    df['longitude'] = -35.96
    df['elevation'] = 2702
    df['method'] = 4
    df['date'] = 20120601
    df['timestamp'] = pd.to_datetime('2012-06-01')
    df['error'] = -9999
    
    # plt.figure()
    # plt.plot(df.density, -df.midpoint, marker ='o')
    
    sumup_index_conflict = check_conflicts(df_sumup, df,
                                            var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
    df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    
    
    df = pd.read_excel('data/density data/Vallelonga/NGRIP-Density-accumulation.xlsx')
    df = df.iloc[6:, 0:2]
    
    df.columns = ['midpoint', 'density']
    df['start_depth'] = df.midpoint - 0.55/2
    df['stop_depth'] = df.midpoint + 0.55/2
    df['density'] = df.density *1000
    
    df['reference_full'] = 'Vallelonga, P., Christianson, K., Alley, R. B., Anandakrishnan, S., Christian, J. E. M., Dahl-Jensen, D., Gkinis, V., Holme, C., Jacobel, R. W., Karlsson, N. B., Keisling, B. A., Kipfstuhl, S., Kjær, H. A., Kristensen, M. E. L., Muto, A., Peters, L. E., Popp, T., Riverman, K. L., Svensson, A. M., Tibuleac, C., Vinther, B. M., Weng, Y., and Winstrup, M.: Initial results from geophysical surveys and shallow coring of the Northeast Greenland Ice Stream (NEGIS), The Cryosphere, 8, 1275–1287, https://doi.org/10.5194/tc-8-1275-2014, 2014.'
    df['reference_short'] = 'Vallelonga et al. (2014)' 
    df['reference'] = ind_ref
    df['profile'] = df_sumup.profile.max() + 1
    df['profile_name'] = 'NGRIP_1999'
    df['latitude'] = 75.10
    df['longitude'] = -42.32
    df['elevation'] = 2917
    df['method'] = 4
    df['date'] = 19990701
    df['timestamp'] = pd.to_datetime('1999-07-01')
    df['error'] = -9999
    
    sumup_index_conflict = check_conflicts(df_sumup, df,
                                            var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
    df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    
    # %% EGIG 2022 cores
    print('loading EGIG 2022 core from J. Harper')
    ref_ind = df_sumup.reference.max() + 1
    for f in os.listdir('data/density data/EGIG 2022 Harper'):
        if '.xml' in f: continue
        if '_den' in f:
            print(f)
            nskip = 5
            if f in ['2022_T3_25m_den.txt','2022_T4_25m_den.txt', 
                     '2022_T3_6m_den.txt']: nskip = 6
            df_meta = pd.read_csv('data/density data/EGIG 2022 Harper/'+f,encoding='mbcs', sep=':', skipinitialspace=True, header = None).iloc[:nskip,:]
    
            df = pd.read_csv('data/density data/EGIG 2022 Harper/'+f,encoding='mbcs', sep='\t', skiprows=nskip)
        
            df.columns = ['start_depth', 'stop_depth', 'mass','density', 'notes']
            df[['start_depth', 'stop_depth', 'density']] = df[['start_depth', 'stop_depth', 'density']].astype(float)
            df[['start_depth', 'stop_depth']] = df[['start_depth', 'stop_depth']]/100
            df['midpoint'] = df.start_depth + (df.stop_depth-df.start_depth)/2
            
            df['reference_full'] = 'Harper, J., & Humphrey, N. (2022). Firn density and ice content at sites along the west Expéditions Glaciologiques Internationales au Groenland (EGIG) line, Greenland, 2022. Arctic Data Center. doi:10.18739/A2KH0F10F.'
            df['reference_short'] = 'Harper and Humphrey (2022)' 
            df['reference'] = ref_ind 
            df['profile'] = df_sumup.profile.max() + 1
            df['profile_name'] = df_meta.iloc[0,1].replace('\t','')
            coords = df_meta.iloc[1,1].replace('\t','').replace('-','').split('(')[0].replace('N','').replace('"','').split('W')
            df['latitude'] = int(coords[0].split(' ')[0]) + float(coords[0].split(' ')[1])/60
            df['longitude'] = -(int(coords[1].split(' ')[0]) + float(coords[1].split(' ')[1])/60)
            df['elevation'] = -9999
            df['method'] = 4
            df['date'] = int(df_meta.iloc[2,1].replace('\t','').replace('-',''))
            df['timestamp'] = pd.to_datetime(df_meta.iloc[2,1].replace('\t',''))
            df['error'] = -9999
            
            # plt.figure()
            # plt.plot(df.density, -df.midpoint, marker ='o')
            
            sumup_index_conflict = check_conflicts(df_sumup, df,
                                                    var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
            df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    # %% EGIG 2018-2019 cores
    print('loading EGIG 2018-2019 core from J. Harper')
    ref_ind = df_sumup.reference.max() + 1
    for f in os.listdir('data/density data/EGIG 2018-2019 Harper'):
        if '.xml' in f: continue
        if '_den.csv' in f:
            nskip = 5
            # if f in ['2022_T3_25m_den.txt','2022_T4_25m_den.txt', 
            #          '2022_T3_6m_den.txt']: nskip = 6
            df_meta = pd.read_csv('data/density data/EGIG 2018-2019 Harper/'+f,encoding='mbcs', skipinitialspace=True, header = None).iloc[:nskip,:]
    
            df = pd.read_csv('data/density data/EGIG 2018-2019 Harper/'+f,encoding='mbcs', skiprows=nskip)
        
            df.columns = ['start_depth', 'stop_depth', 'mass','density', 'notes']
            df[['start_depth', 'stop_depth', 'density']] = df[['start_depth', 'stop_depth', 'density']].astype(float)
            df[['start_depth', 'stop_depth']] = df[['start_depth', 'stop_depth']]/100
            df['midpoint'] = df.start_depth + (df.stop_depth-df.start_depth)/2
            
            df['reference_full'] = 'Harper, J., & Humphrey, N. (2023). Firn density and ice content at sites along the west EGIG line, Greenland, 2018 and 2019. Arctic Data Center. doi:10.18739/A2QB9V701.'
            df['reference_short'] = 'Harper and Humphrey (2023)' 
            df['reference'] = ref_ind 
            df['profile'] = df_sumup.profile.max() + 1
            df['profile_name'] = df_meta.iloc[0,1].replace(' ','')
            coords = df_meta.iloc[1,1].replace('N','')
            
            df['latitude'] = int(coords.split(' ')[0]) + float(coords.split(' ')[1])/60
            coords = df_meta.iloc[1,2].replace('W','')
            df['longitude'] = -(int(coords.split(' ')[0]) + float(coords.split(' ')[1])/60)
            df['elevation'] = -9999
            df['method'] = 4
            df['timestamp'] = pd.to_datetime(df_meta.iloc[2,1], format='%m/%d/%Y')
            df['date'] = df['timestamp'].dt.strftime('%Y%m%d').astype(int)
            df['error'] = -9999
            
            # plt.figure()
            # plt.plot(df.density, -df.midpoint, marker ='o')
            
            print(df[['profile_name','timestamp','latitude','longitude']].drop_duplicates().values)
            sumup_index_conflict = check_conflicts(df_sumup, df,
                                                    var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
            df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

    # %% Reed_DYE-2
    print('loading Reed_DYE-2')
    
    df = pd.read_excel('data/density data/Reed_DYE-2/Dye-2 density profile.xlsx', skiprows=2)
    df.columns = ['midpoint', 'start_depth', 'stop_depth', 'density']
    
    df['reference_full'] = 'Reed, S. 1966. Performance studyof the Dewlin ice cap stations Greenland, 1963. Cold Regions Research and Engineering Laboratory. Special Report 72.'
    df['reference_short'] = 'Reed (1966)' 
    df['reference'] = df_sumup.reference.max() + 1
    df['profile'] = df_sumup.profile.max() + 1
    df['profile_name'] = 'DYE-2'
    df['latitude'] = 66.4726
    df['longitude'] = -46.282983
    df['elevation'] = 2119
    df['method'] = 4
    df['date'] = 19630416
    df['timestamp'] = pd.to_datetime('1963-04-16')
    df['error'] = -9999
    plt.figure()
    plt.plot(df.density, -df.midpoint, marker ='o')
    
    sumup_index_conflict = check_conflicts(df_sumup, df,
                                            var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
    df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    
    # %% Camp century climate
    print('loading Camp Century Climate')
    df = pd.read_excel('data/density data/Camp Century Climate/Camp_Century_snow_firn_density.xlsx',
                       sheet_name='CoreB62')
    df = df.iloc[:,[0, 1, 3]]
    df.columns = ['start_depth', 'stop_depth', 'density']
    
    df['midpoint'] = df.start_depth + (df.stop_depth - df.start_depth)/2
    df['reference_full'] =  'Colgan, W., Pedersen, A., Binder, D., Machguth, H., Abermann, J., & Jayred, M. (2018). Initial field activities of the Camp Century Climate Monitoring Programme in Greenland. GEUS Bulletin, 41, 75-78. https://doi.org/10.34194/geusb.v41.4347. Data: Colgan, W., Camp Century: Firn density measurements in cores B73 and B62, https://doi.org/10.22008/FK2/UFGONU, GEUS Dataverse, V1, 2021'
    df['reference_short'] = 'Colgan et al. (2018), Colgan (2021)' 
    df['reference'] = df_sumup.reference.max() + 1
    df['profile'] = df_sumup.profile.max() + 1
    df['profile_name'] = 'Camp Century (Core B62)'
    df['latitude'] = 77.1714
    df['longitude'] = -61.0778
    df['elevation'] = 1887
    df['method'] = 4
    df['date'] = 20170728
    df['timestamp'] = pd.to_datetime('2017-07-28')
    df['error'] = -9999
    df.loc[(df.midpoint>40)&(df.density<200), 'density'] = np.nan
    plt.figure()
    plt.plot(df.density, -df.midpoint, marker ='o')
    
    sumup_index_conflict = check_conflicts(df_sumup, df,
                                            var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
    df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)

    # %% adding N. Clerx cores
    df_meta = pd.read_excel('data/density data/Clerx data/firn_cores_2021.xlsx')
    ind_ref = df_sumup.reference.max() + 1
    for c in df_meta.core:
        print(c)
        df = pd.read_excel('data/density data/Clerx data/firn_cores_2021.xlsx', sheet_name = c)
        df.columns = [v.replace(' ','_').replace('(','').replace(')','') for v in df.columns]
        df_formatted = df.groupby('density').depth_cm.min().rename('start_depth').reset_index()
        df_formatted['stop_depth'] = df.groupby('density').depth_cm.max()
        df_formatted['midpoint'] = df_formatted.start_depth + (df_formatted.stop_depth - df_formatted.start_depth)/2
        df_formatted['profile'] = df_sumup.profile.max() + 1
        df_formatted['profile_name'] = c
        df_formatted['reference'] = ind_ref
        df_formatted['reference_short'] = 'Clerx et al. (2022)'
        df_formatted['method'] = 4
        df_formatted['timestamp'] = df_meta.loc[df_meta.core==c, 
                                                'datetime cored (UTC)'].dt.strftime('%Y-%m-%d').values[0]
        df_formatted['date'] = int(df_meta.loc[df_meta.core==c, 
                                               'datetime cored (UTC)'].dt.strftime('%Y%m%d').values[0])
        df_formatted['latitude'] = df_meta.loc[df_meta.core==c, 'N'].values[0]
        df_formatted['longitude'] = df_meta.loc[df_meta.core==c, 'E'].values[0]
        df_formatted['elevation'] = df_meta.loc[df_meta.core==c, 'Z'].values[0]
        df_formatted['error'] = np.nan
        
        df_formatted['reference_full'] = 'Clerx, N., Machguth, H., Tedstone, A., Jullien, N., Wever, N., Weingartner, R., and Roessler, O.: In situ measurements of meltwater flow through snow and firn in the accumulation zone of the SW Greenland Ice Sheet, The Cryosphere, 16, 4379–4401, https://doi.org/10.5194/tc-16-4379-2022, 2022. Data: Clerx, N., Machguth, H., Tedstone, A., Jullien, N., Wever, N., Weingartner, R., and Roessler, O. (2022). DATASET: In situ measurements of meltwater flow through snow and firn in the accumulation zone of the SW Greenland Ice Sheet [Data set]. In The Cryosphere. Zenodo. https://doi.org/10.5281/zenodo.7119818'
        # sumup_index_conflict = check_conflicts(df_sumup, df_formatted,
        #                                         var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
        df_sumup = pd.concat((df_sumup, df_formatted[necessary_variables]), ignore_index=True)
    # %% Kawakami Greenland SE Dome core
    print('adding Kawakami Greenland SE Dome core')
    ind_ref = df_sumup.reference.max()+1
    for k, f in enumerate(os.listdir("data/density data/Kawakami/")):
        df = pd.read_csv("data/density data/Kawakami/"+f,
                         sep='\t', skiprows=114, encoding ='mbcs')
        df[df==-999] = np.nan
        
        df.columns =  ['midpoint','2', 'density']       
        half_thick = (df.midpoint.values[1:] - df.midpoint.values[:-1])/2
        half_thick = np.append(half_thick, half_thick[-1])
        df['start_depth'] = df.midpoint - half_thick
        df['stop_depth'] = df.midpoint + half_thick
       
        df['reference_full'] = " Kawakami, K., Iizuka, Y., Sasage, M., Matsumoto, M., Saito, T., Hori, A., et al. (2023). SE-Dome II ice core dating with half-year precision: Increasing melting events from 1799 to 2020 in southeastern Greenland. Journal of Geophysical Research: Atmospheres, 128, e2023JD038874. https://doi.org/10.1029/2023JD038874. Data: Kawakami, K.; Iizuka, Y. (2023-08-11): NOAA/WDS Paleoclimatology - SE-Dome ll Ice Core, South Eastern Greenland Accumulation Rate, Melt Crust and Feature, H2O2 and Tritium Concentration, Bulk Density and Electrical Conductivity Data from 1800 to 2020 CE. NOAA National Centers for Environmental Information. https://doi.org/10.25921/bx51-ng14."
        df['latitude'] = 67.19
        df['longitude'] = -36.47
        df['elevation'] = 3160.7
        
        df['timestamp'] ='2021-06-01'
        df['date'] = df['timestamp'].str.replace('-','').astype(int)
        if k == 0:
            df['profile_name'] = 'SE-Dome II (bulk)'
            df['method'] = 4
        else:
            df['profile_name'] = 'SE-Dome II (X-ray)'
            df['method'] = 10
        df['error'] = np.nan
        df['profile'] = df_sumup.profile.max()+1
        df['reference'] = ind_ref
        df['reference_short'] = "Kawakami et al. (2023)"
        # print(df[['profile_name','date', 'latitude','longitude','elevation', 'reference_short']].drop_duplicates().values)
        # sumup_index_conflict = check_conflicts(df_sumup, df,
        #     var=['profile', 'profile_name', 'date', 'start_depth', 'density'])
        df_sumup = pd.concat((df_sumup, df[necessary_variables]), ignore_index=True)
    # %% return final df
    return df_sumup