from redis import Redis

NIKE_URL = [r"https://www.nike.com/w/graphql?queryid=products&endpoint=%2Fproduct_feed%2Frollup_threads%2Fv2%3Ffilter%3Dmarketplace(US)%26filter%3Dlanguage(en)%26filter%3DemployeePrice(true)%26filter%3DattributeIds(16633190-45e5-4830-a068-232ac7aea82c)%26anchor%3D",
            r"%26count%3D",
            r"%26consumerChannelId%3Dd9a5bc42-4b9c-4976-858a-f159cf99c647"]
NUM_ITEMS = 1309
ITEMS_PER_PAGE = 24

def make_url(page):
    start = page * ITEMS_PER_PAGE
    return "".join([NIKE_URL[0], str(start), NIKE_URL[1], str(ITEMS_PER_PAGE), NIKE_URL[2]])

def main():
    redis = Redis()
    redis.lpush("nike:start_urls", *[make_url(i) for i in range((NUM_ITEMS // ITEMS_PER_PAGE) + 1)])

if __name__ == "__main__":
    main()
