# -*- coding: utf-8 -*-

# clean_types.py dicates the datatype for each column
# it uses the MultiLabelBinarizer(mlb) to encode categorical data
# it also standardises the values within each column,
# merging columns created by the mlb where required

'df_details_c'
'df_synopsis_c'
'df_cast_c'
'df_awards_c'
'df_table_c'
'df_source_c'

load_folder = 'clean_cols'
save_folder = 'clean_types'

import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Clean_df import Clean_df
import datetime
from dateutil.parser import parse

pd.set_option('display.max_rows', 5000)
clean = Clean_df(load_folder, save_folder)

# Processing df_details
def process_df_details():
    df_name = 'df_details'
    df = clean.load_df(df_name)
    
    # Processing 'Genre' column
    column = 'Genre'
    df_mlb = clean.create_mlb(df, column, [' ', '/'])
    genre_merge_list = {'comedy': '.*com.*', 
                        'drama': '.*(drama|web).*',
                        'romance': '.*roman.*',
                        'investigative': '.*investigat.*',
                        'sci-fi': '.*sci.*',
                        'thriller': '.*(thriller|suspense).*',
                        'time travel': '.*time.*',
                        'music': '.*music.*',
                        'political': 'politic',
                        'period': '.*period.*',
                        'sports': '.*sport.*',
                        'mystery': '.*m[iy]ster.*',
                        }
    clean.merge_cols(df_mlb, genre_merge_list, is_mlb=True)
    
    genre_del_list = ['^$',
                      '&',
                      ]
    genre_del_list += [col for col in df_mlb if df_mlb[col].sum() < 2]
    clean.del_cols(df_mlb, genre_del_list)
    
    return clean.merge_mlb(df, df_mlb, column)
    
    # Processing 'Related_show_merged' column
    column = 'Related_show_merged'
    df[column] = df[column].notnull()
    
    # Processing 'Airtime_merged' column
    column = 'Airtime_merged'
    
    regex_split_list = {'Airtime': '(\d.*)',
                        'Airdays': '(\D*)(?:\s\d|$)',#(.*days?|[.])',
                        }
    
    clean.split_cols(df, 'Airtime_merged', regex_split_list)
    
    # Processing 'Airdays' column
    column = 'Airdays'
    
    regex_replace_list = {'Mon': 'Mon\S*',
                        'Tue': 'Tue\S*',
                        'Wed': 'Wed\S*',
                        'Thur': 'Thur\S*',
                        'Fri': 'Fri\S*',
                        'Sat': 'Sat\S*',
                         'Sun': 'Sun\S*',
                        'Mon to Fri': '.*Weekdays.*',
                        'Mon to Sun': '.*(Everyday|each).*',
                        }
    
    clean.replace_values_in_col(df, column, regex_replace_list)
    
    # Split string to work with word sets
    df[column] = df[column].str.split()
    
    # Remove all elements not in the keys set
    def only_keys(x):
        keys = ('Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun', 'to')
        try:
            return [ele for ele in x if ele in keys]
        except TypeError:
            return []
    
    df[column] = df[column].apply(only_keys)
    
    # Convert ['Mon', 'to', 'Thur'] to ['Mon', 'Tue', 'Wed', Thur']
    def expand_to(x):
        days = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
        try:
            index = x.index('to')
            start = days.index(x[index-1])
            end = days.index(x[index+1])
            if start > end:
                return days[start:] + days[:end+1]
            else:
                return days[start:end+1]
        except ValueError:
            return x
            
    df[column] = df[column].apply(expand_to)
    
    # Add 'No info' to list if it is empty
    def fill_no_info(x):
        if not x:
            return ['No_info']
        return x
    
    df[column] = df[column].apply(fill_no_info)
    
    # Create mlb
    df_mlb = clean.create_mlb(df, column, is_list=True)
    
    # Merge mlb back with main df
    df = clean.merge_mlb(df, df_mlb, column)
    
    # Processing 'Airtime' column
    column = 'Airtime'
    
    regex_delete_list = ['\(.+?\)',
                         'changed to',
                         '.*\)',
                         'to.*',
                         '-\d+:\d*',
                         '~.*'
                         ]
    
    clean.delete_values_in_col(df, column, regex_delete_list)
    
    # Standardise PM in caps, change 24 to 00 and . to : for parse to work
    regex_replace_list = {'PM': 'pm',
                          '00': '24',
                          ':': '.',
                          }
    
    clean.replace_values_in_col(df,column, regex_replace_list)
    
    # Filter out all time values and [Pp][Mm]
    def filter_time_values(x):
        return re.findall('(\d+:?\d*|PM)', str(x))
    
    df[column] = df[column].apply(filter_time_values)
    
    # Set all time values to datetime.time
    def set_time_type(x):
        
        is_pm = False
        try:
            x.remove('PM')
            is_pm = True
        except ValueError:
            pass
        times = [parse(t) for t in x]
        if is_pm:
            pm = datetime.timedelta(hours=12)
            times = [t+pm for t in times if t.hour < 13]
        return [t.time() for t in times]
    
    df[column] = df[column].apply(set_time_type)
    
    # Plot time graph to determine how to bin
    # explode() to get all time data and compute bins
    data = df[column].explode().dropna()
    
    def in_hours(x):
        return x.hour+(x.minute/60)
    data = data.apply(in_hours)
    
    #for nbin in range(4,24,4):
    #    plt.figure(nbin, figsize=(12,5))
    #    plt.title(nbin)
    #    bins = np.interp(np.linspace(0, len(data), nbin + 1),
    #                     np.arange(len(data)),
    #                     np.sort(data))
    #    bins = [round(b) for b in bins]
    #    plt.xticks(bins)
    #    plt.hist(data, bins)
    
    nbin = 12
    bins = np.interp(np.linspace(0, len(data), nbin + 1),
                     np.arange(len(data)),
                     np.sort(data))
#    labels = list(range(nbin))
    
    #for nbin in range(4,24,4):
    #    plt.figure(nbin+24, figsize=(12,5))
    #    plt.title(nbin)
    #    bins = pd.qcut(data, nbin, precision=1, duplicates='drop')
    #    plt.hist(data)
    
    # split list data into multiple columns
    df2 = pd.DataFrame(df[column].values.tolist(), index=df.index)
    
    # for each column
    for col in df2.columns:
    
        # remove all empty rows
        df2[col].dropna(inplace=True)
        df2[col] = df2[col].apply(in_hours)
        
        # apply pd.cut() to all relevant columns
        x = pd.cut(df2[col].tolist(),
                   bins,
                   duplicates='drop')
        x.categories = [1,2,3,4,5,6,7,8,9]
        df2[col] = x
        
    # combine all relevant columns via index
    df[column] = df2.values.tolist()
    
    def remove_nan(x):
        return list(filter(lambda v: v==v, x))
    df[column] = df[column].apply(remove_nan)
    
    # fill in missing rows with 'No info'
    def add_noinfo(x):
        if not x:
            x.append('No_info')
        return x
    df[column] = df[column].apply(add_noinfo)
    
    # Create mlb
    df_mlb = clean.create_mlb(df,
                              column,
                              is_list=True,
                              is_str=False)
    
    df = clean.merge_mlb(df, df_mlb, column)
    
    # Processing 'Names_merged' column
    column = 'Names_merged'
    
    # Get number of unique names as str
    def len_as_str(x):    
        # replace / with , and split by ,
        x = x.replace('/', ',').split(',')
        
        # strip and lower
        x = [t.strip().lower() for t in x]
        return str(len(set(x)))
    
    df[column] = df[column].apply(len_as_str)
    
    # Create mlb and merge
    df_mlb = clean.create_mlb(df, column)
    print(df_mlb.columns)
    df = clean.merge_mlb(df, df_mlb, column)
    
    # Processing Period_merged column
    column = 'Period_merged'
    
    def find_dates(x):
        if x:
            return re.findall('\d{4}-\w+-\d{2}', x)
        return []
    
    df[column] = df[column].apply(find_dates)
    
    df2 = pd.DataFrame(df[column].values.tolist(), index=df.index)
    
    #df2['lenlist'] = df[column].apply(len)
    df2.columns = ['Period_start',
                   'Period_end',
                   'Period_rerun_start',
                   'Period_rerun_end']
    
    df2['Period_end'].fillna(df2['Period_start'],
                             inplace = True)
    
    # Convert type from str to datetime
    def to_datetime(x):
        if x:
            return parse(x)
        return None
    
    df2 = df2.applymap(to_datetime)
    
    del df[column]
    df = pd.concat([df, df2], axis=1)
    
    # To use categorical embedding later
    # hash on year?
    
    # Processing 'Network_merged' column
    column = 'Network_merged'
    
    df_mlb = clean.create_mlb(df, column, delimiter=['/', '&', 'and'])
    
    regex_merge_list = {'kbs1': 'kbs1',
                        'kbs2': 'kbs2',
                        'kbs-n': 'kbs-?n',
                        'mnet': 'mnet',
                        'mbc plus': 'mbc ',
                        'oksusu': 'oksusu',
                        'dramax': 'dramax',
                        }
    clean.merge_cols(df_mlb, regex_merge_list, is_mlb=True)
    
    # Remove all single instance categories
    regex_del_list = [col for col in df_mlb if df_mlb[col].sum() < 2]
    clean.del_cols(df_mlb, regex_del_list)
    
    df_mlb.rename(columns={'(aired at yunsae university main auditorium)': 'yunsae'},
                  inplace=True)
    
    df = clean.merge_mlb(df, df_mlb, column)
    
    # Processing 'Song_merged' column
    column = 'Song_merged'
    
    df[column] = df[column].astype('bool')
    
    # Remove 'Rating_merged' column
    column = 'Rating_merged'
    del df[column]
    
    # Processing 'Episodes_merged' column
    column = 'Episodes_merged'
    
    def clean_episodes(x):
        if x:
            reg = '\s*(\d+)\s*(\+\s*(\d*)\s*)?(?:\(.*\))?'
            r = re.compile(reg)
            x = x.split('or')
            has_match = False
            for i, ele in enumerate(x):
                m = r.match(ele)
                if m:
                    has_match = True
                    if m.group(3):
                        x[i] = int(m.group(1)) + int(m.group(3))
                    else:
                        x[i] = int(m.group(1))
            if has_match:
                return x
        return None
        
    df[column] = df[column].apply(clean_episodes)
    
    # Save df_details
    clean.save_df(df)
    
# Processing df_cast
def process_df_cast():
    df_name = 'df_cast'
    df = clean.load_df(df_name)
    
    # Processing 'cast_type' column
    column = 'cast_type'
    
    # Set three cast types (Main, Supporting, Others)
    # Change all .*family.* to Supporting
    regex_replace_list = {'Main Cast': '[Mm]ain.*',
                          'Supporting Cast': '.*([Ff]amily|[Ss]up+ort|[Aa]round).*',
                          'Others': '^((?!Main|Support).)*$',
                          }
    clean.replace_values_in_col(df, column, regex_replace_list)
    
    # Save df_cast
    clean.save_df(df)

# Processing df_awards
def process_df_awards():
    df_name = 'df_awards'
    df = clean.load_df(df_name)
    
    # Remove 'award_type' column
    column = 'award_type'
    del df[column]
    
    # Processing 'awardees' column
    column = 'awardees'
    df[column] = df[column].apply(lambda x: x.split('(')[0])
    
    # Save df_awards
    clean.save_df(df)

# Processing df_table
def process_df_table():
    df_name = 'df_table'
    df = clean.load_df(df_name)
    
    def clean_ratings(x):
        if x:
            reg = '\D*(\d+\.\d?).*'
            r = re.compile(reg)
            m = r.match(x)
            if m:
                return float(m.group(1))
        return None
    
    # Process each rating column
    # 'TNmS Nationwide', 'TNmS Seoul', 'AGB Nationwide', 'AGB Seoul'
    for column in ['TNmS Nationwide',
                   'TNmS Seoul',
                   'AGB Nationwide',
                   'AGB Seoul']:
        df[column] = df[column].apply(clean_ratings)
    
    # Check number of episode ratings vs number of episode total
    # Groupby drama, count(), mean()
    rating_count = df.groupby('drama').count()
    del rating_count['Date']
    rating_count.rename(columns = lambda x: x+'_count', inplace=True)
    rating_mean = df.groupby('drama').mean()
    rating_mean.rename(columns = lambda x: x+'_mean', inplace=True)
    
    df = pd.concat([rating_count, rating_mean], axis=1)
    # Redeclare df.name
    df.name = df_name
    df.reset_index(level=0, inplace=True)
    
    # Save df_table
    clean.save_df(df)

# Processing df_source
def process_df_source():
    df_name = 'df_source'
    df = clean.load_df(df_name)
    
    # Save df_source
    clean.save_df(df)

# Processing df_synopsis
def process_df_synopsis():
    df_name = 'df_synopsis'
    df = clean.load_df(df_name)
    
    # Save df_synopsis
    clean.save_df(df)


#print(df.info())
#print(list(df)) # Print columns
#print(df.iloc[0, :]) # Print first row
print('-----------------------------------')

'''
df_details
    Run 'Genre' through the MLB then through clean_cols
    Related Drama and Related Series - Has Related Show (Boolean)
    Airtime_merged - Split days (mlb) and timing slot (bin)
    Names_merged - Num alternate names (mlb)
    Period_merged - Split start and end (day, month, year)
    Network_merged - (mlb)
    Song_merged - Has song (song hits at start and middle of period?)
    Rating_merged - Remove
    Episodes_merged - Split by ( and remove last, split by + and add all nums
    
df_cast
    cast_type - Main, supporting, others (supporting = family)
        Combine with replace() regex
    
df_awards
    Check when awards are given/nominated
    award_type - Remove
    awardees - str.split('(')[0] to get only the awardee name
    * awardee is the drama itself if awardee == drama
    
df_table
    drama - filter all dramas without ratings (search for rating site)
    Date - Ratings only started 2012-05-26?
    
df_source
    No current use
df_synopsis
    No current use
'''

process_df_details()

process_df_cast()

process_df_awards()

process_df_table()

process_df_source()

process_df_synopsis()