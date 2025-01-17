# -*- coding: utf-8 -*-

# For data cleaning before uploading to database
import pandas as pd
import json

# url title info table
# info = {segments}

# Create the relevant DataFrames
df_details = None
df_synopsis = None
df_cast = None
df_awards = None
df_table = None
df_source = None

# Functions to process data
def details(data, title):
    
    global df_details
    ddict = {}
    ddict['drama'] = title
    for line in data.split('\n'):
        line = line.strip()
        detail, value = line.split(':', 1)
        detail = detail.strip()
        value = value.strip()
        ddict[detail] = value
    df_temp = pd.DataFrame(ddict, index=[0])
    if df_details is None:
        df_details = df_temp
    else:
        df_details = pd.concat([df_details, df_temp])
    df_details.name = 'df_details'

def synopsis(data, title):
    
    global df_synopsis
    sdict = {'synopsis': data,
             'drama': title,
             }
    if df_synopsis is None:
        df_synopsis = pd.DataFrame(sdict, index=[0])
    else:
        row = df_synopsis.index[df_synopsis['drama'] == title]
        if not row.empty:
            df_synopsis.loc[row, 'synopsis'] += '\n %s' % data
        else:
            df_synopsis = df_synopsis.append(sdict, ignore_index=True)
            
    df_synopsis.name = 'df_synopsis'

# put into different columns depending on cast_type    
cast_type = 'Main Cast' # default
current_title = ''

def cast(data, title):
    
    global df_cast, cast_type, current_title
    castings = data.split('\n')
    # check if data contains cast_type or actors
    data_type = castings[0].split(' as ')
    if len(data_type) < 2:
        cast_type = data_type[0].strip()
        current_title = title
    else:
        if title != current_title:
            cast_type = 'Main Cast'
            current_title = title
        cdict = {'cast_type': cast_type,
                 'drama': title,
                 }
        actors = []
        characters = []
        for casting in castings:
            values = casting.split(' as ')
            actors.append(values[0].strip())
            if len(values) > 1:
                characters.append(values[1].strip())
            else:
                characters.append('-')
        cdict['actors'] = actors
        cdict['characters'] = characters
        df_temp = pd.DataFrame(cdict)
        if df_cast is None:
            df_cast = df_temp
        else:
            df_cast = pd.concat([df_cast, df_temp])
        df_cast.name = 'df_cast'

def awards(data, title):
    
    global df_awards
    award_list = data.split('\n')
    df_temp = None
    for award in award_list:
        split_1 = award.strip().split(' : ')
        split_2a = split_1[0].strip().split(' ', 1)
        award_year = split_2a[0].strip()
        award_name = split_2a[1].strip()
        split_2b = split_1[1].split(' - ')
        award_type = split_2b[0].strip()
        split_3 = split_2b[1].split(' & ')
        awardees = [awardee.strip() for awardee in split_3]
        # create seperate df first for each award to ensure proper broadcasting
        tempdict = {'award_year': award_year,
                    'award_name': award_name,
                    'award_type': award_type,
                    'awardees': awardees,
                    'drama': title,
                    }
        award_temp = pd.DataFrame(tempdict, index=range(len(awardees)))
        if df_temp is None:
            df_temp = award_temp
        else:
            df_temp = pd.concat([df_temp, award_temp])
    if df_awards is None:
        df_awards = df_temp
    else:
        df_awards = pd.concat([df_awards, df_temp])
    df_awards.name = 'df_awards'

def table(data, title):
    
    global df_table
    header_rows = 0
    rows = []
    row = []
    for element in data:
        if element == '\n\t\t':
            continue
        if element == '\n\t':
            if header_rows == 0:
                if len(row) == 6:
                    header_rows = 2
                else:
                    header_rows = 1
            rows.append(row)
            row = []
            continue
        row.append(element)
    columns = [i + ' ' + j for i, j in zip(*rows[:header_rows])]
    df_temp = pd.DataFrame(rows[header_rows:], columns=columns)
    df_temp['drama'] = title
    if df_table is None:
        df_table = df_temp
    else:
        df_table = pd.concat([df_table, df_temp])
    df_table.name = 'df_table'

def sources(data, title):
    
    global df_source
    split = data.split(': ')
    if len(split) < 2 or 'source' not in split[0].lower():
        return
    sdict = {'source': data,
             'drama': title,
             }
    df_temp = pd.DataFrame(sdict, index=[0])
    if df_source is None:
        df_source = df_temp
    else:
        df_source = pd.concat([df_source, df_temp])
    df_source.name = 'df_source'

# Get list of all segment types
#seg_list = {}

# Parsing through the Scrape.jl file
try:
    with open('Scrape.jl', 'r', encoding='utf-8') as f:
        for drama in f:
            data = json.loads(drama)
            title = data['url'][0]
            if 'title' in data:
                title = data['title'][0]
            print('Processing %s' % title)
            for segment_name, segment_data in data['info'][0].items():
                segment_type = segment_name.split()[0].lower()
                
                # Get list of segment_types
#                if segment_type in seg_list:
#                    seg_list[segment_type].append(title)
#                else:
#                    seg_list[segment_type] = [title]
                
                # Cleaning sources segment
                if segment_type == 'episode':
                    segment_type = 'sources'
                
                # Cleaning cast segment
                def set_cast_type(cast_type):
                    if segment_data:
                        try:
                            globals()['cast'](cast_type, title)
                        except:
                            pass
                
                if segment_type == 'main':
                    segment_type = 'cast'
                    set_cast_type('Main Cast')
                if segment_type in ['support', 'supporting']:
                    segment_type = 'cast'
                    set_cast_type('Support')
                if segment_type in ['cameos', 'other', 'others', 'extended']:
                    segment_type = 'cast'
                    set_cast_type('Others')
                if segment_data: # Only process non-empty data segments
                    try:
                        globals()[segment_type](segment_data, title)
                        #print('%s segment processed' % segment_type)
                    except:
                        pass
                        #print('Segment_type: %s not found' % segment_type)
            if 'table' in data:
                try:
                    # process table
                    globals()['table'](data['table'], title)
                    #print('table segment processed')
                except:
                    pass
                    #print('Segment_type: table not found')
            else:
                pass
                #print('Table not found')
except FileNotFoundError:
    print('Scrapy.jl not found')

#print('List of all segment types: ')
#for seg, titles in seg_list.items():
#    print(seg, len(titles))

# print all dfs and save to feather
import feather
for df in [df_details,
           df_synopsis,
           df_cast,
           df_awards,
           df_table,
           df_source,
           ]:
    print(df)
    print('           ----------          ')
    if df is None:
        print(df)
        continue
    globals()[df.name].reset_index(drop=True,
                                   inplace=True,
                                   ) # feather does not save the index (req)
    feather.write_dataframe(globals()[df.name],
                            './feather/%s' % df.name,
                            )
    print('Saved to feather')
    print('           ----------          ')