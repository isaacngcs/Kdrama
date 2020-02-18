# -*- coding: utf-8 -*-

# clean_cols.py removes all columns that are unnecessary to the dataset
# it also merges data that should be categorised under the same column

'df_details'
'df_synopsis'
'df_cast'
'df_awards'
'df_table'
'df_source'

load_folder = 'feather'
save_folder = 'clean_cols'

from Clean_df import Clean_df
clean = Clean_df(load_folder, save_folder)

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
                      'Related_show_merged': '.*Related.*',
                      }
    
clean.process_df('df_details', details_del_list, details_merge_list)

clean.process_df('df_synopsis')

clean.process_df('df_cast')

clean.process_df('df_awards')

table_merge_list = {'Episode': '[Ee][Pp].*',
                    'TNmS Seoul': 'TNmS\s[ST].*',
                    'Date': 'Date',
                    }

clean.process_df('df_table', [], table_merge_list)

clean.process_df('df_source')