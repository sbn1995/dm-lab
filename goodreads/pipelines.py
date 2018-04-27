# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from scrapy import signals
import json
from .spiders.GoodreadsSpider import GoodreadsSpider

class GoodreadsPipeline(object):
    def open_spider(self, spider):
        filename = 'g_'+str(GoodreadsSpider.user_id)+'_'+str(GoodreadsSpider.pages[0])+'_'+str(GoodreadsSpider.pages[1])+'.jl'
        self.file = open(filename, 'w')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item
