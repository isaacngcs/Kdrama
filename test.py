# -*- coding: utf-8 -*-

import feather

df = feather.read_dataframe('./clean_cols/df_details_c')

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

mlb = MultiLabelBinarizer()
df['Genre'] = df['Genre'].fillna('').apply(lambda x: x.split(',')) \
                .apply(lambda xlist: [x.strip().lower() for x in xlist])
df2 = pd.DataFrame(mlb.fit_transform(df['Genre']), columns=mlb.classes_)

print(df)
print(df2)
print(mlb.classes_)
'''
TO DO:
    MultiLabelBinarizer
    https://chrisalbon.com/machine_learning/preprocessing_structured_data/one-hot_encode_features_with_multiple_labels/
    
    Run 'Genre' through the MLB then through clean_cols
    
    
    
    
    
    
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