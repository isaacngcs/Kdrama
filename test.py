# -*- coding: utf-8 -*-

import pandas as pd
'''
url = 'https://en.wikipedia.org/wiki/Extraordinary_You'
tables = pd.read_html(url)

for table in tables:
    print('  ---   ')
    print(table)
    


df[['a', 'b', 'c', 'd']] = df['1'].str.split(r'(\d+)\s(\w+)', expand=True)

print(df)
'''
a = {'1': [['1', '2', '3'], ['a', 'b', 'c'], ['z', 'z', 'z']],
     '2': ['e', 'f', 'g']}

df = pd.DataFrame(a)

print((df['1'].map(type) == list).all())

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