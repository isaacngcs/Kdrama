# -*- coding: utf-8 -*-

import feather
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

class Clean_df():
    
    def __init__(self, load_folder, save_folder):
        
        self.load_folder = load_folder
        self.save_folder = save_folder
        
    def load_df(self, name, load_folder=None):
        
        if load_folder is None:
            load_folder = self.load_folder
        df = feather.read_dataframe('./%s/%s' % (load_folder, name))
        df.name = name
        print('%s loaded from feather' % df.name)
        print('           ----------          ')
        return df
    
    def save_df(self, df, save_folder=None):
        
        if save_folder is None:
            save_folder = self.save_folder
        df.reset_index(drop=True,
                       inplace=True,
                       )
        feather.write_dataframe(df,
                                './%s/%s' % (save_folder, df.name),
                                )
        print('%s saved to feather' % df.name)
        print('           ----------          ')
    
    def del_cols(self, df, regex_del_list):
        
        col_list = list(df)
        for regex in regex_del_list:
            r = re.compile(regex)
            print('For regex %s:' % regex)
            for col in filter(r.match, col_list):
                print('  %s deleted' % col)
                del df[col]
                
    def merge_cols(self, df, regex_merge_list, is_mlb=False):
        
        col_list = list(df)
        for merged, regex in regex_merge_list.items():
            r = re.compile(regex)
            print('For regex %s:' % regex)
            cols = list(filter(r.match, col_list))
            print('  %s merged' % cols)
#            df[cols] = df[cols].replace(r'^\s*$', None, regex=True) # np.nan?
            if is_mlb:
                df[merged] = df[cols].eq(1).any(1)
            else:
                df[merged] = df[cols].apply(lambda x: ', '.join(x.dropna()), axis=1)
            df[merged] = df[merged].replace(r'^\s*$', np.nan, regex=True)
            for col in cols:
                if col != merged:
                    del df[col]
    
    def split_cols(self, df, column, regex_split_list):
        for col, regex in regex_split_list.items():
            df[col] = df[column].str.extract(regex)
            print('Created column %s from regex %s on column %s:' % (col,
                                                                     regex,
                                                                     column))
        del df[column]
    
    def replace_values_in_col(self,
                              df,
                              col,
                              regex_replace_list,
                              is_list=False):
        if is_list:
            for new, reg in regex_replace_list.items():
                df[col] = df[col].apply(lambda xlist: [re.sub(reg, new, x) for x in xlist])
        else:
            for new, reg in regex_replace_list.items():
                df[col] = df[col].str.replace(reg, new, regex=True)
                
    def delete_values_in_col(self,
                             df,
                             col,
                             regex_delete_list,
                             is_list=False):
        if is_list:
            for reg in regex_delete_list:
                df[col] = df[col].apply(lambda xlist: [x for x in xlist if not re.match(reg, x)])
        else:
            for reg in regex_delete_list:
                df[col] = df[col].str.replace(reg, '', regex=True)
    
    def process_for_model(self,
                          df,
                          col,
                          delimiter=',',
                          is_list=False,
                          regex_replace_list=None,
                          regex_delete_list=None):
        if not is_list:
            df[col] = df[col].fillna('')
            for d in delimiter:
                df[col] = df[col].apply(lambda x: x.replace(d, ','))
            df[col] = df[col].apply(lambda x: x.split(','))
        df[col] = df[col].apply(lambda xlist: [str(x).strip().lower() for x in xlist])
                         
        # If regex_replace_list or regex_del_list present
        if regex_replace_list:
            self.replace_values_in_col(df,
                                       col,
                                       is_list=True,
                                       regex_replace_list=regex_replace_list)
        if regex_delete_list:
            self.delete_values_in_col(df,
                                      col,
                                      is_list=True,
                                      regex_delete_list=regex_delete_list)
        
        df[col] = df[col].apply(lambda xset: {x for x in xset}) \
                         .apply(list)
    
    def create_mlb(self,
                   df,
                   col,
                   delimiter=',',
                   is_list=False,
                   regex_replace_list=None,
                   regex_del_list=None): 
        self.process_for_model(df, 
                               col, 
                               delimiter, 
                               is_list, 
                               regex_replace_list,
                               regex_del_list)
        
        mlb = MultiLabelBinarizer()
        
        df_col = pd.DataFrame(mlb.fit_transform(df[col]), columns=mlb.classes_)
        return df_col.astype('bool')
        
    def merge_mlb(self, df, mlb, original_column):
        
        del df[original_column]
        # Rename all columns to start with original column
        mlb.rename(columns = lambda x: original_column+'_'+x, inplace=True)
        return pd.concat([df, mlb], axis=1)
        
    def process_df(self,
                   df_name,
                   del_list=[],
                   merge_list={},
                   load_folder=None,
                   save_folder=None,
                   ):
        
        # Load second df, df_synopsis
        if load_folder is None:
            load_folder = self.load_folder
        df = self.load_df(df_name, load_folder)
        
        self.del_cols(df, del_list)
        self.merge_cols(df, merge_list)
        
        print('Columns: ')
        print(list(df)) # Print columns
        print('\nFirst row of df: ')
        print(df.iloc[0, :]) # Print first row
        print('\ndf info: ')
        print(df.info())
        print('-----------------------------------')
        
        # Save df_detail to clean_df folder
        if save_folder is None:
            save_folder = self.save_folder
        self.save_df(df, save_folder)