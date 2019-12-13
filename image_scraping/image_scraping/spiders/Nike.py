import json
from scrapy.spiders import Rule
from image_scraping.items import ShoeImage
from scrapy.linkextractors import LinkExtractor
from scrapy_redis.spiders import RedisCrawlSpider


class Nike(RedisCrawlSpider):
    """Spider that gets image urls from Nike product pages"""
    name = "nike"
    redis_key = "nike:start_urls"
    allowed_domains = ["nike.com"]

    def parse(self, response):
        json_response = json.loads(response.body_as_unicode())

        if "data" in json_response:
            json_response = json_response["data"]
        if "products" in json_response:
            json_response = json_response["products"]

        for product in json_response["objects"]:
            item = ShoeImage()
            item["image_urls"] = [product["publishedContent"]["properties"]["productCard"]["properties"]["squarishURL"]]
            yield item
