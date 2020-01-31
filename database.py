# -*- coding: utf-8 -*-

# For uploading to database
import pandas as pd
import feather

# May reduce memory usage by using 'with feather.read_dataframe(...) as df

df_list = ['df_details',
           'df_synopsis',
           'df_cast',
           'df_awards',
           'df_table',
           'df_source',
           ]

for df in df_list:
    globals()[df] = feather.read_dataframe('./feather/%s' % df)
    globals()[df].name = df
    print('%s loaded from feather' % df)
    print('           ----------          ')

# Method 1
# Normal Method
from sqlalchemy import create_engine
engine = create_engine('postgresql+psycopg2://username:password@host:port/database',
                       echo=False,
df.to_sql(name_the_table, con = engine)

# Method 2
# Insert data directly into postgres table:
'''
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
'''