# -*- coding: utf-8 -*-

# For uploading to database
import feather
import re

'df_details'
'df_synopsis'
'df_cast'
'df_awards'
'df_table'
'df_source'

load_folder = 'feather'
save_folder = 'clean_cols'

def load_df(name):
    df = feather.read_dataframe('./%s/%s' % (load_folder, name))
    df.name = name
    print('%s loaded from feather' % df.name)
    print('           ----------          ')
    return df

def save_df(df):
    df.reset_index(drop=True,
                   inplace=True,
                   )
    feather.write_dataframe(df,
                            './%s/%s_c' % (save_folder, df.name),
                            )
    print('%s saved to feather' % df.name)
    print('           ----------          ')

def del_cols(df, regex_del_list):
    col_list = list(df)
    for regex in regex_del_list:
        r = re.compile(regex)
        print('For regex %s:' % regex)
        for col in filter(r.match, col_list):
            print('  %s deleted' % col)
            del df[col]

def merge_cols(df, regex_merge_list):
    col_list = list(df)
    for merged, regex in regex_merge_list.items():
        r = re.compile(regex)
        print('For regex %s:' % regex)
        cols = list(filter(r.match, col_list))
        print('  %s merged' % cols)
        df[merged] = df[cols].apply(lambda x: ', '.join(x.dropna()), axis=1)
        for col in cols:
            if col != merged:
                del df[col]

def process_df(df_name, del_list=[], merge_list={}):
    
    # Load second df, df_synopsis
    df = load_df(df_name)
    
    del_cols(df, del_list)
    merge_cols(df, merge_list)
    
    print(list(df)) # Print columns
    print(df.iloc[0, :]) # Print first row
    print(df.info())
    print('-----------------------------------')
    
    # Save df_detail to clean_df folder
    save_df(df)

details_del_list = ['.*\(Part.*',
                    'Tagline',
                    'Series',
                    'Format',
                    'Language',
                    'Related\sTV\s[Ss]hows?',
                    'Released\sdate',
                    ]
        
details_merge_list = {'Airtime_merged': '.*Air\s?[Tt]ime.*',
                      'Names_merged': '.*([Aa]lso|[Pp]reviou|[Tt]it*le|Name|known).*',
                      'Period_merged': '.*adcast\s[Ppdy].*',
                      'Network_merged': 'Broadcas($|t\s[Nnv].*)',
                      'Song_merged': '.*(Theme|[Ss]ong|Insert).*',
                      'Rating_merged': 'Viewership\sratings?',
                      'Episodes_merged': 'Ep(s$|isode.*)',
                      }
    
process_df('df_details', details_del_list, details_merge_list)

process_df('df_synopsis')

process_df('df_cast')

process_df('df_awards')

table_merge_list = {'Episode': '[Ee][Pp]\s.*',
                    'TNmS Seoul': 'TNmS\s[ST].*',
                    }

process_df('df_table', [], table_merge_list)

process_df('df_source')