import json
from scrapy.spiders import Rule
from shoe_images.items import ShoeImage
from scrapy.linkextractors import LinkExtractor
from scrapy_redis.spiders import RedisCrawlSpider


class Nike(RedisCrawlSpider):
    """Spider that gets image urls from Nike product pages"""
    name = "nike"
    redis_key = "nike:start_urls"
    allowed_domains = ["nike.com"]

    def parse(self, response):
        json_response = json.loads(response.body_as_unicode())

        for product in json_response["data"]["products"]["objects"]:
            item = ShoeImage()
            item["image_urls"] = product["publishedContent"]["properties"]["productCard"]["properties"]["squarishURL"]
            yield item
        




"""
Nike url:
https://www.nike.com/w/graphql?queryid=products&anonymousId=C7200C4604FA91CC1071C156D4B95DFC&endpoint=/product_feed/rollup_threads/v2?filter=marketplace(US)&filter=language(en)&filter=employeePrice(true)&filter=attributeIds(0f64ecc7-d624-4e91-b171-b83a03dd8550,16633190-45e5-4830-a068-232ac7aea82c)&anchor=72&count=24&consumerChannelId=d9a5bc42-4b9c-4976-858a-f159cf99c647
"""
