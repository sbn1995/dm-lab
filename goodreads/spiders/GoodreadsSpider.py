# -*- coding: utf-8 -*-
import scrapy
from ..items import GoodreadsItem

USER = 'JANA' #change this to your name
PAGE_START_END = [1,100] #and this to the pages you want to scrape
user_dict = {'JANA':0, 'ROC':1, 'SABIN':2}

class GoodreadsSpider(scrapy.Spider):
    pages = PAGE_START_END
    name = 'Goodreads'
    allowed_domains = ['goodreads.com', 'images.gr-assets.com']
    base_page = "http://www.goodreads.com/book/show/"
    handle_httpstatus_list = [404]
    user_id = user_dict[USER]
    num_users = len(user_dict)
    page = pages[0] + user_id
    start_urls = ['http://www.goodreads.com/book/show/'+str(page)]

#//*[@id="moreBookData"]/table[2]/tbody/tr[1]/td[2]
    def parse(self, response):
        n = self.page
        print('Page:'+str(n))
        if not response.status == 404:
            img = response.xpath('//img[@id="coverImage"]/@src').extract_first()
            goodreads_item = GoodreadsItem()

            box_set = False
            box_str = ['Boxed', 'Boxed Set', 'Box Set']
            box_details = [st in response.xpath('//*[@id="details"]/div[1]').extract_first() for st in box_str]
            box_name = [st in response.xpath('//*[@id="bookTitle"]/text()').extract_first().strip() for st in box_str]
            if(True in box_details or True in box_name):
                box_set = True
            #num_votes = int(response.xpath('//*[@id="bookMeta"]/a[2]/span/text()').extract_first().strip().replace(',',''))
            e_info = self.get_edition_info(response.xpath('//*[@id="bookMeta"]/script/text()').extract_first())
            num_votes = e_info[1][1]

            #yield goodreads_item
            if img and not box_set and num_votes > 100:
                goodreads_item['author'] = response.xpath('//*[@id="bookAuthors"]/span[2]/a[1]/span/text()').extract_first()
                goodreads_item['name'] = response.xpath('//*[@id="bookTitle"]/text()').extract_first().strip()
                try:
                    goodreads_item['release_date'] = response.xpath('//*[@id="details"]/div[2]/text()').extract_first().split('\n')[2].strip()
                except:
                    pass
                desc = response.xpath('//*[@id="description"]/span[2]/text()').extract()
                if(not desc):
                    desc = response.xpath('//*[@id="description"]/span[1]/text()').extract_first()
                goodreads_item['description'] = desc
                goodreads_item['num_votes_all_editions'] = e_info[1][1]
                goodreads_item['avg_rating_all_editions'] = e_info[1][0]
                goodreads_item['added_by_all_editions'] = e_info[1][3]
                goodreads_item['added_by_this_edition'] = e_info[0][3]
                goodreads_item['to_reads'] = e_info[1][4]
                goodreads_item['avg_rating_this_edition'] = e_info[0][0]
                goodreads_item['num_votes_this_edition'] = e_info[0][1]
                try:
                    goodreads_item['num_pages'] =  int(response.css('span[itemprop="numberOfPages"]::text').extract_first().split()[0])
                except:
                    pass
                goodreads_item['series'] =  'yes' if response.xpath('//*[@id="bookTitle"]/a').extract_first() else ''
                try:
                    goodreads_item['isbn'] = response.xpath('//*[@id="bookDataBox"]/div[2]/div[2]/text()').extract_first().strip()
                except:
                    pass
                goodreads_item['top_genre'] = response.css('a[class="actionLinkLite bookPageGenreLink"]::text').extract_first()
                goodreads_item['page'] = n
                goodreads_item['image_urls'] = [img]
                yield goodreads_item

        self.page += self.num_users
        next_page = self.base_page + str(n+self.num_users)
        if self.page < self.pages[1]:
            yield scrapy.Request(next_page, callback=self.parse, dont_filter = True)


    def get_edition_info(self, whole_text):
        m = whole_text[whole_text.find("\"")+1:]
        this_edition_list = m[m.lower().find("this edition"):m.lower().find("goodreads")].split('\\n')
        all_editions_list = m[m.lower().find("all editions"):m.lower().find("this edition")].split('\\n')
        this_edition = []
        for i in range(2,6):
            tmp = this_edition_list[i]
            tmp = tmp[tmp.find(">")+1:]
            tmp = tmp[:tmp.find("<")]
            if(i == 2):
                this_edition.append(float(tmp))
            else:
                this_edition.append(int(tmp))
        all_editions = []
        for i in range(2,7):
            tmp = all_editions_list[i]
            tmp = tmp[tmp.find(">")+1:]
            tmp = tmp[:tmp.find("<")]
            if(i == 2):
                all_editions.append(float(tmp))
            else:
                all_editions.append(int(tmp))
        #[this[rating, numratings, reviews, added], all[rating, numratings, reviews, added, toread]]
        return [this_edition,all_editions]
