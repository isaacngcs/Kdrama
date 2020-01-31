# -*- coding: utf-8 -*-
import scrapy
from Kdrama.items import KdramaItem
from scrapy.loader import ItemLoader

class ScrapeSpider(scrapy.Spider):
    
    name = 'Scrape'
    custom_settings = {
            'ITEM_PIPELINES': {
                    'Kdrama.pipelines.DuplicatesPipeline': 250,
                    'Kdrama.pipelines.SaveFilePipeline': 251,
                    'Kdrama.pipelines.ProcessImagesPipeline': 300,
                    #'Kdrama.pipelines.DatabasePipeline': 301,
                    }
            }
        
    def get_requests(self):
        
        import json
        items = set([])
        links = set([])
        crawled = 'Crawl'
        scraped = 'Scrape'
    
        try:
            with open('%s.jl' % crawled, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                for link in loaded['links']:
                    links.add(link)
            print('%s.jl imported' % crawled)
        except FileNotFoundError:
            print('%s.jl not found' % crawled)
            return links
      
        try:
            with open('%s.jl' % scraped, 'r', encoding='utf-8') as f:
                for line in f:
                    items.add(json.loads(line)['url'][0])
            print('%s.jl imported' % scraped)
        except IOError:
            print('%s.jl not found' % scraped)
        
        unscraped = []
        if items:
            unscraped = list(links - items)    
        else:
            unscraped = links
        print('Links to scrape: %d out of %d' % (len(unscraped), len(links)))
        return unscraped
        
    def start_requests(self):
        
        # Test single page
        #links = ['http://www.koreandrama.org/angels-last-mission-love/',
        #         'http://www.koreandrama.org/wanna-taste/']
        
        for link in self.get_requests():
            yield scrapy.Request(
                url = link,
                callback = self.parse_product,
                meta={'source_url': link,
                      },
            )

    def parse_product(self, response):
        
        from unidecode import unidecode
        # use unidecode on all fields
        def decode(data):
            if isinstance(data, list):
                return [decode(x) for x in data]
            return unidecode(data) if data else data
        
        l = ItemLoader(item = KdramaItem(),
                       response = response,
                       )
        l.add_value('url', response.meta['source_url'])
        l.add_css('title', decode('h3.post-title a ::text'))
        
        info = response.css('div.entrytext p')
        segments = {}
        part_id = 1
        segment = 'Details'
        for p in info:
            new_segment = decode(p.css('strong ::text').extract_first())
            if new_segment:
                segment = new_segment
                part_id = 1
            else:
                segment_part = segment + ' ' + str(part_id)
                segments[segment_part] = ' '.join(decode(p.css('::text').extract()))
                part_id = part_id + 1        
        l.add_value('info', segments)
        
        l.add_css('table', decode('table tr ::text'))
        
        yield l.load_item()