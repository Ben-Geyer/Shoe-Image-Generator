from redis import Redis

URL_START = r"https://www.nike.com/w/graphql?queryid=products&endpoint=%2Fproduct_feed%2Frollup_threads%2Fv2%3Ffilter%3Dmarketplace(US)%26filter%3Dlanguage(en)%26filter%3DemployeePrice(true)%26filter%3D"
ATTR_IDS_MEN = r"attributeIds(0f64ecc7-d624-4e91-b171-b83a03dd8550%2C16633190-45e5-4830-a068-232ac7aea82c)"
ATTR_IDS_WOMEN = r"attributeIds(16633190-45e5-4830-a068-232ac7aea82c%2C7baf216c-acc6-4452-9e07-39c2ca77ba32)"
URL_END = [r"%26anchor%3D",
           r"%26count%3D",
           r"%26consumerChannelId%3Dd9a5bc42-4b9c-4976-858a-f159cf99c647"]
NUM_ITEMS_MEN = 725
NUM_ITEMS_WOMEN = 475
ITEMS_PER_PAGE = 24

def make_url(page, attr_ids):
    start = page * ITEMS_PER_PAGE
    return "".join([URL_START, attr_ids, URL_END[0], str(start), URL_END[1], str(ITEMS_PER_PAGE), URL_END[2]])

def main():
    redis = Redis()
    redis.lpush("nike:start_urls", *[make_url(i, ATTR_IDS_MEN) for i in range((NUM_ITEMS_MEN // ITEMS_PER_PAGE) + 1)])
    redis.lpush("nike:start_urls", *[make_url(i, ATTR_IDS_WOMEN) for i in range((NUM_ITEMS_WOMEN // ITEMS_PER_PAGE) + 1)])

if __name__ == "__main__":
    main()
