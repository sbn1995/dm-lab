# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from collections import OrderedDict

class OrderedItem(scrapy.Item):
    def __init__(self, *args, **kwargs):
        self._values = OrderedDict()
        if args or kwargs:  # avoid creating dict for most common case
            for k, v in six.iteritems(dict(*args, **kwargs)):
                self[k] = v

class GoodreadsItem(OrderedItem):
    # define the fields for your item here like:
    # name = scrapy.Field()
    #file_urls = scrapy.Field()
    author = scrapy.Field()
    name = scrapy.Field()
    release_date = scrapy.Field()
    num_votes_all_editions = scrapy.Field()
    avg_rating_all_editions = scrapy.Field()
    num_votes_this_edition = scrapy.Field()
    avg_rating_this_edition = scrapy.Field()
    added_by_this_edition = scrapy.Field()
    added_by_all_editions = scrapy.Field()
    to_reads = scrapy.Field()
    num_pages = scrapy.Field()
    series =  scrapy.Field()
    isbn = scrapy.Field()
    top_genre = scrapy.Field()
    page = scrapy.Field()
    image_urls = scrapy.Field()
    images = scrapy.Field()
    description = scrapy.Field()
