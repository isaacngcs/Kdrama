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

from Clean_df import Clean_df
clean = Clean_df(load_folder, save_folder)

# Processing df_details
df_name = 'df_details'
df = clean.load_df(df_name)
'''
# Processing 'Genre' column
column = 'Genre'
df_col = clean.create_mlb(df, column, [' ', '/'])
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
clean.merge_cols(df_col, genre_merge_list, is_mlb=True)

genre_del_list = ['^$',
                  '&',
                  ]
genre_del_list += [col for col in df_col if df_col[col].sum() < 2]
clean.del_cols(df_col, genre_del_list)

clean.merge_mlb(df, df_col, column)

# Processing 'Related_show_merged' column
column = 'Related_show_merged'
df[column] = df[column].notnull()
'''
# Processing 'Airtime_merged' column
column = 'Airtime_merged'
#print(df[column].unique())

#m = re.search(r, string)
#print(m.group(1))

#!!!!!!!!!!!!!'Get the split groups'
#(.*(days?|[.]))\s(\d{1,2}[.:]\d{2})
regex_split_list = {'Airtime': '(\d.*)',
                    'Airdays': '(\D*)(?:\s\d|$)',#(.*days?|[.])',
                    }

clean.split_cols(df, 'Airtime_merged', regex_split_list)

print(df[df['Airdays'].isnull()])
print(df['Airtime'])

df['Airdays'] = df['Airdays'].str.split()
print(df['Airdays'])

regex_convert_list = {'Mon': 'Mon.*',
                      'Tue': 'Tue.*',
                      'Wed': 'Wed.*',
                      'Thur': 'Thur.*',
                      'Fri': 'Fri.*',
                      'Sat': 'Sat.*',
                      'Sun': 'Sun.*',
                      }

days = ('Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun')

#clean.replace_cols(df, 'Airdays', regex_convert_list)



#convert airdays to mlb
'''

Replace current with shorthand:
    Mon, tue, wed, thur, fri, sat, sun

'to' takes all in between days

'''




print(df.info())
print(list(df)) # Print columns
#print(df.iloc[0, :]) # Print first row
print('-----------------------------------')

'''
df_details
    Run 'Genre' through the MLB then through clean_cols
    Related Drama and Related Series - Has Related Show (Boolean)
    Airtime_merged - Split days (mlb) and timing slot (bin)
    Names_merged - Num alternate names (redo merge, how many different langs)
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
    
df_table
    drama - filter all dramas without ratings (search for rating site)
    Date - Ratings only started 2012-05-26?
'''

#process_df('df_details')

#process_df('df_synopsis')
#
#process_df('df_cast')
#
#process_df('df_awards')
#
#process_df('df_table')
#
#process_df('df_source')