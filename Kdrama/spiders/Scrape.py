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
                    'Kdrama.pipelines.DatabasePipeline': 301,
                    }
            }
        
    def get_requests(self):
        
        import json
        items = []
        links = []
        crawled = 'Crawl'
        scraped = 'Scrape'
    
        try:
            with open('%s.jl' % crawled, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                for link in loaded['links']:
                    links.append(link)
            print('%s.jl imported' % crawled)
        except FileNotFoundError:
            print('%s.jl not found' % crawled)
            return links
      
        try:
            with jsonlines.open('%s.jl' % scraped, 'r') as f:
                for line in f:
                    items.append(line['url'])
            print('%s.jl imported' % scraped)
        except IOError:
            print('%s.jl not found' % scraped)
        
        if items:
            unscraped = list(set(links) - set(items))    
        else:
            unscraped = links
        print('Links to scrape: %d out of %d' % (len(unscraped), len(links)))
        return unscraped
        
    def start_requests(self):
        
        # Test single page
        links = ['http://www.koreandrama.org/angels-last-mission-love/']
        
        for link in links:#self.get_requests():
            yield scrapy.Request(
                url = link,
                callback = self.parse_product,
                meta={'source_url': link,
                      },
            )

    def parse_product(self, response):
        
        l = ItemLoader(item = KdramaItem(),
                       response = response,
                       )
        l.add_value('url', response.meta['source_url'])
        l.add_css('title', 'h3.post-title a ::text')
        
        info = response.css('div.entrytext p')
        segments = {}
        part_id = 1
        segment = 'Details'
        for p in info:
            new_segment = p.css('strong ::text').extract_first()
            if new_segment:
                segment = new_segment
                part_id = 1
            else:
                segments[segment + str(part_id)] = ' '.join(p.css('::text').extract())
                part_id = part_id + 1        
        l.add_value('info', segments)
        
        table_loader = l.nested_css('table')
        table_loader.add_css('table', 'tr ::text')
        
        yield l.load_item()