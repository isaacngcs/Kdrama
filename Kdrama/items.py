# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class LinksItem(scrapy.Item):
    
    url = scrapy.Field()
    links = scrapy.Field()
    count = scrapy.Field()

class KdramaItem(scrapy.Item):
    
    url = scrapy.Field()
    title = scrapy.Field()
    info = scrapy.Field()
    table = scrapy.Field()
    
    '''
    title = scrapy.Field()
    genre = scrapy.Field()
    num_episodes = scrapy.Field()
    network = scrapy.Field()
    broadcast_period = scrapy.Field()
    air_slot = scrapy.Field()
    synopsis = scrapy.Field()
    cast = scrapy.Field()
    '''