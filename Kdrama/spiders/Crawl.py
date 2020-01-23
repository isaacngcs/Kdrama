# -*- coding: utf-8 -*-
import scrapy
#from scrapy_splash import SplashRequest
from Kdrama.items import LinksItem

class CrawlSpider(scrapy.Spider):
    name = 'Crawl'
    custom_settings = {
            'ITEM_PIPELINES': {
                    'Kdrama.pipelines.SaveFilePipeline': 251,
                    }
            }
    
    def start_requests(self):
        start_url = 'https://www.http://www.koreandrama.org/list-of-korean-drama/'
        
        yield scrapy.Request(url = start_url,
                             callback = self.parse,
                             meta = {'source_url': start_url,
                                     },
                             )
        
        # check if SplashRequest args are in Request args (timeout wait resource_timeout)
        
        yield SplashRequest(
            url = start_url,
            callback = self.parse,
            args={'timeout': 3600,
                  'wait': 0.5,
                  'resource_timeout': 20,
                  },
            meta={'source_url': start_url,
                  }
        )

    def parse(self, response):
        base_url = 'https://www.appliancesonline.com.au'
        
        item = LinksItem()
        item['url'] = response.meta['source_url']
        item['links'] = set([])
        
        index = 0
        for product in response.css('aol-product.product'):
            link =  product.css('a ::attr(href)').extract_first()
            if link.split('/')[1] != 'product':
                continue
            print('LINK FOUND: ' + str(index) + '. ' + link)
            index = index + 1
            item['links'].add(base_url + link)    
        # Convert to list so it is JSON serializable
        item['links'] = list(item['links'])
        item['count'] = index
        
        yield item