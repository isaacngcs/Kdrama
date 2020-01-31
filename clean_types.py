# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# For uploading to database
import feather
import re

'df_details_c'
'df_synopsis_c'
'df_cast_c'
'df_awards_c'
'df_table_c'
'df_source_c'

load_folder = 'clean_cols'
save_folder = 'clean_types'

def load_df(name):
    df = feather.read_dataframe('./%s/%s_c' % (load_folder, name))
    df.name = name
    print('%s loaded from feather' % df.name)
    print('           ----------          ')
    return df

def save_df(df):
    df.reset_index(drop=True,
                   inplace=True,
                   )
    feather.write_dataframe(df,
                            './%s/%s_t' % (save_folder, df.name),
                            )
    print('%s saved to feather' % df.name)
    print('           ----------          ')

def multilabel(df, col):
    df[col] = df[col].fullna('')
    df[col] = df[col].apply

def process_df(df_name):
    
    # Load second df, df_synopsis
    df = load_df(df_name)
    
    col = 'Genre'
    df[col] = df[col].fillna('')
    df[col] = df[col].apply(lambda x: x.split(','))
    
    print(list(df)) # Print columns
    print(df.iloc[0, :]) # Print first row
    print(df.info())
    print('-----------------------------------')
    
    # Save df_detail to clean_df folder
    save_df(df)

process_df('df_details')

#process_df('df_synopsis')
#
#process_df('df_cast')
#
#process_df('df_awards')
#
#process_df('df_table')
#
#process_df('df_source')