FROM python:3.7-slim-buster
COPY ./image_scraping/requirements.txt /
RUN pip3 install -r /requirements.txt
COPY ./image_scraping /
ENTRYPOINT ["scrapy"]
CMD ["crawl", "nike"]
