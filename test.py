# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import datetime
from dateutil.parser import parse

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.max_colwidth = 500


'''
url = 'https://en.wikipedia.org/wiki/Extraordinary_You'
tables = pd.read_html(url)

for table in tables:
    print('  ---   ')
    print(table)
    


df[['a', 'b', 'c', 'd']] = df['1'].str.split(r'(\d+)\s(\w+)', expand=True)

print(df)
'''

a = {'data1': ['1992', '2002', '2020', None, '1999'],
     'data2': ['b', 'c', '1', '2', 'five']}

df = pd.DataFrame(a)

column = 'actors'

df[column] = ['-  Yoon Seo',
                'Kim Sang Soon (gimsangsun)',
                'Lee Tae Yeon (itaeyeon)',
                'Kim Mi kyung',
                '-  Park Cho Rong']

column = 'lists'
df[column] = [['a', 'b'],
              ['b', 'a'],
              ['a', '2'],
              ['a'],
              ['b'],
              ]

print(df[column])

column2 = 'lists2'
df[column2] = [['1', '2'],
               ['1', '3'],
               ['2', '3'],
               ['1', '2', '3'],
               [],
               ]

from sklearn.feature_extraction import FeatureHasher

n_features = 5
enc = FeatureHasher(n_features=n_features,
                    input_type='string')
encoded = enc.transform(df[column]).toarray()
print(pd.DataFrame(encoded, columns=[column+'_'+str(x) for x in range(n_features)]))

encoded = enc.transform(df[column2]).toarray()
print(encoded)

# df[column] = df[column].apply(lambda xlist: [x for x in xlist if not re.match('a', x)])

# print(df[column])

# a = 'a,,b'
# a2 = a.split(',')
# print(a2)

# print(re.match('a', a))

# column = ['actors', 'lists']
# print(df[column])

#reg = '(\w+(\s\w+)*)'
#r = re.compile(reg)
#string = 'Jung Hyun Suk (jeonghyeonseog)'
#
#m = r.match(string)
#print(m.group(0))
'''

import feather
folder = 'clean_cols'
file = 'df_table'

df = feather.read_dataframe('./%s/%s' % (folder, file))

col = 'drama'
#df = df.loc[df[col] == '']
#for _, row in df.iloc[:20].iterrows():
#    print(row)

import pandas as pd
#df[col] = pd.to_datetime(df[col])

#print(df.sort_values(by=[col], inplace=True, ascending=True))
print(list(df))
print(df.info())

print(df[col].unique())
print(len(df[col].unique()))
'''

'''
TO DO:
    MultiLabelBinarizer
    https://chrisalbon.com/machine_learning/preprocessing_structured_data/one-hot_encode_features_with_multiple_labels/
    
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
    
    
    
    list of wikipedia pages,
    to extract:
        summary data at right of page
    
    
    
    
    Setup database
    Only add if not duplicate (or overwrite)
    
    Normal transfer method:
    
        from sqlalchemy import create_engine
        engine = create_engine('sqlite://', echo=False)
        df.to_sql(name_the_table, con = engine)
    
    Insert data directly into postgres table:
        
        engine = create_engine('postgresql+psycopg2://username:password@host:port/database')
    
        df.head(0).to_sql('table_name', engine, if_exists='replace',index=False) #truncates the table
        
        conn = engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        contents = output.getvalue()
        cur.copy_from(output, 'table_name', null="") # null values become ''
        conn.commit()
    
    
    
    Ratings:
        2 Different agencies
        Public vs Paid (Cable)
        Google trends
            Find absolute value if possible
            Else use a benchmark average kdrama, possibly by year?
'''