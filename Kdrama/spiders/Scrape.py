# -*- coding: utf-8 -*-
import scrapy


class ScrapeSpider(scrapy.Spider):
    name = 'Scrape'
    allowed_domains = ['www.koreandrama.org']
    start_urls = ['http://www.koreandrama.org/']

    def parse(self, response):
        pass
