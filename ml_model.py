# -*- coding: utf-8 -*-

# Setup model for training

import numpy as np
import pandas as pd
from Clean_df import Clean_df

from functools import reduce
# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 500

# Start with a test model
# Average rating prediction

'----- Import data -----'

# Import data
'df_details'
'df_cast'
'df_awards'
'df_table'

load_folder = 'clean_types'
save_folder = 'ml_model'

clean = Clean_df(load_folder, save_folder)

df_details = clean.load_df('df_details')
df_cast = clean.load_df('df_cast')
df_awards = clean.load_df('df_awards')
df_table = clean.load_df('df_table')

'----- df_details -----'

#print(df_details.info(verbose=True))

# Set the minimum number of times a genre has to appear
# min_count = 2
# regex_del_list += [col for col in df_mlb if df_mlb[col].sum() < min_count]
    
# Set minimum number of instances for networks
# min_instances = 2
# regex_del_list = [col for col in df_mlb if df_mlb[col].sum() < min_instances]
# clean.del_cols(df_mlb, regex_del_list)
    
'Find all features that need reconversion back into lists or just mlb with min_instances'

'----- df_cast -----'

# Get top n number of actors according to number of appearances
column = 'actors'
n = 50000
cutoff = df_cast.groupby(column).apply(len).nlargest(n).min() - 1

# Drop actors with appearances one below the lowest of the top n
df_cast = df_cast.groupby(column).filter(lambda x: len(x) > cutoff)

# Group all cast by drama into lists by actors or actors+cast_type
# Setting up actors_cast_type column
df_cast['actors_cast_type'] = df_cast['cast_type'] + ':' + df_cast['actors']

# Choose which column to use, actors or actors_cast_type
#filter_col = 'actors'
#to_drop = 'actors_cast_type'
filter_col = 'actors_cast_type'
to_drop = 'actors'

df_cast.drop(columns=to_drop)

# Filter with chosen filter_col
df_cast = df_cast.groupby(['drama'])[filter_col].apply(list).reset_index()

# Create mlb
df_mlb = clean.create_mlb(df_cast,
                          col=filter_col,
                          is_list=True)
#print('temp.. df_mlb: ', df_mlb.shape)

# Combine mlb
df_cast = clean.merge_mlb(df_cast, df_mlb, filter_col)

'----- df_awards -----'

# Not to use, need to prevent any link of 'future' information
# How to link only previous data?

'----- df_table -----'

# Choose which rating to analyse
rating_cols = ['TNmS Nationwide_mean',
               'TNmS Seoul_mean',
               'AGB Nationwide_mean', 
               'AGB Seoul_mean',
               ]

rating = rating_cols[0]

cols = ['drama', rating]

df_table = df_table[cols]

# Rename rating column
df_table = df_table.rename({rating: 'rating'}, axis=1)

#print(df_table.info())

'----- Merge into single df -----'

# Check shape of dfs
print('df_details: ', df_details.shape)
print('df_cast: ', df_cast.shape)
print('df_awards: ', df_awards.shape)
print('df_table: ', df_table.shape)

data = None

dfs = [df_details,
       df_cast,
#       df_awards,
       df_table,
       ]

data = reduce(lambda x, y: pd.merge(x, y, on = 'drama'), dfs)
#print(data.info(verbose=True))

# Remove all rows without df_table data
data = data[data['rating'].notna()]
#print(data.info())


'Print all columns in ml_model'
# print(data.columns)

# print(df_cast.info())




'----- Save ml_model -----'

data.name = 'data'
clean.save_df(data)