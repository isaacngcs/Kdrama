# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import scrapy
from scrapy.exceptions import DropItem  
from scrapy.pipelines.images import ImagesPipeline
from scrapy.exporters import JsonLinesItemExporter

identifier = '' # Enter identifier

class DuplicatesPipeline(object): 
    
    def __init__(self): 
        self.identifier_seen = set() 

    def process_item(self, item, spider): 
        if item[identifier] in self.identifier_seen: 
            raise DropItem("Repeated item found: %s" % item) 
        else: 
            self.identifier_seen.add(item[identifier]) 
            return item

class ProcessImagesPipeline(ImagesPipeline):
    
    def file_path(self, request, response=None, info=None):
        product_name = request.meta['product_name']
        image_type = request.meta['image_type']
        image_name = request.meta['image_name']
        return '%s/%s/%s.jpg' % (product_name, image_type, image_name)

    def get_media_requests(self, item, info):
        product_name = item['product_name']
        for num, image_url in item['image_urls'].items():
            yield scrapy.Request(url=image_url,
                                 meta = {'product_name': product_name,
                                         'image_type': num.split()[0],
                                         'image_name': num})

class SaveFilePipeline(object):
    
    def __init__(self):
        self.files = {}
        
    # Have a meta file to check new items against Scrape.jl
    # New entries append when duplicates are skipped
    
    '''
    Two files, one for item data, one for scraped items
    Scrape.jl = item data file
    Links_done.jl = scraped items file
    
    Links_done first accessed though spider first to get list
    Links_done accessed again in close_spider to append new scraped items
    '''
    
    def open_spider(self, spider):
        file = open('%s.jl' % spider.name, 'ab+')
        self.files[spider] = file
        self.exporter = JsonLinesItemExporter(file)
        self.exporter.start_exporting() 
        print('Identifier is %s' % identifier)
        
    def close_spider(self, spider):
        self.exporter.finish_exporting()
        file = self.files.pop(spider)
        file.close()
        print('Exporter closed.....')
        
    def process_item(self, item, spider):
        if spider.name == 'Crawl' or item[identifier] is not None:
            self.exporter.export_item(item)
        return item