# -*- coding: utf-8 -*-
import scrapy
#from scrapy_splash import SplashRequest
from Kdrama.items import LinksItem
from scrapy.loader import ItemLoader

class CrawlSpider(scrapy.Spider):
    
    name = 'Crawl'
    custom_settings = {
            'ITEM_PIPELINES': {
                    'Kdrama.pipelines.SaveFilePipeline': 251,
                    }
            }
    
    def start_requests(self):
        
        start_url = 'https://www.koreandrama.org/list-of-korean-drama/'
        
        yield scrapy.Request(url = start_url,
                             callback = self.parse,
                             meta = {'source_url': start_url,
                                     },
                             )
        
    def parse(self, response):
        
        l = ItemLoader(item = LinksItem(),
                       response = response,
                       )
        l.add_value('url', response.meta['source_url'])
        for n in [1, 2]:
            l.add_css('links', 'div.entrytext p:nth-child(%s) a ::attr(href)' % n)
        l.add_value('count', len(l.get_collected_values('links')))
        
        yield l.load_item()