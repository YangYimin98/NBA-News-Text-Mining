import requests
from time import sleep
import math
import os
import ujson
import re
from lxml import html

METADATA_FILE = "/meta/meta.txt"
ARTICLE_PATH = "/article_raw/content"
ARTICLE_BATCH_SIZE = 200


def article_metadata_scraper(batch_size, metadata_file):
    """
    This function exists to scrape the metadata of NBA news from the official website.
    The metadata api is: "https://content-api-prod.nba.com/public/1/content"

    :param batch_size: news batch size of each 'GET' request
    :param metadata_file: relative path of metadata storage
    :return: list, metadata
    """
    api_url = "https://content-api-prod.nba.com/public/1/content"
    max_news_index = 10000
    max_page = math.ceil(max_news_index / batch_size)
    metadata_path = os.getcwd() + metadata_file
    print('Metadata folder: {}'.format(metadata_path))

    metadata = []
    for page_index in range(max_page):
        api_request_param = {
            'page': page_index + 1,
            'count': batch_size,
            'types': 'post',
            'region': 'international-lpp'
        }
        res = requests.get(api_url, params=api_request_param).json()
        for _article in res['results']['items']:
            article_id = _article['id']
            link = _article['permalink']
            date = _article['date'][:10]
            title = _article['title']
            metadata.append({'id': article_id, 'link': link, 'date': date, 'title': title})
        print('Metadata index: {}'.format(len(metadata)))
        sleep(2)
    j_meta = ujson.dumps(metadata)
    with open(metadata_path, 'w') as f:
        f.write(j_meta)
    return metadata


def article_content_scraper(metadata_file, content_file, batch_size=200):
    """
    This function exists to scrape news contents.

    :param metadata_file: relative path of metadata storage
    :param content_file: relative path of content storage
    :param batch_size: content storage size of each file
    :return: None
    """

    # scrape article meta data
    metadata_path = os.getcwd() + metadata_file
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f_meta:
            meta = ujson.load(f_meta)
            print('Meta data loaded: {}'.format(metadata_path))
    else:
        meta = article_metadata_scraper(batch_size=100, metadata_file=metadata_path)

    # scrape article content
    content_path = os.getcwd() + content_file
    file_idx = 0
    batch_idx = 0
    batch_content = []
    for i_meta in meta:
        # scrape content
        print(i_meta)
        para = html.fromstring(requests.get(i_meta['link']).content).xpath(
            '//div[@class="Article_article__2Ue3h"]/p//text()')
        content = ' '.join(para)
        content = re.sub('  ', ' ', re.sub('\n|\r|\t|\xa0', ' ', content))
        print(content)
        i_meta['content'] = content
        batch_content.append(i_meta)
        batch_idx += 1
        if batch_idx == batch_size:
            # save
            with open(content_path + str(file_idx) + '.txt', 'w') as fcw:
                fcw.write(ujson.dumps(batch_content))
                batch_content = []
                file_idx += 1
                batch_idx = 0
        sleep(1.5)


def article_content_check(content_file):
    """
    This function exists to check the content of each article and clean data.

    :param content_file: relative path of content storage
    :return: None
    """
    content_path = os.getcwd() + content_file

    # check data
    file_idx = 0
    while os.path.exists(content_path + str(file_idx) + '.txt'):
        print('Checking file index: {}'.format(file_idx))
        with open(content_path + str(file_idx) + '.txt', 'r') as f_check:
            f_content = ujson.load(f_check)
        for a_idx, a_dict in enumerate(f_content):
            if len(a_dict['content']) < 10:
                print(a_dict)
                para = html.fromstring(requests.get(a_dict['link']).content).xpath(
                    '//div[@class="Article_article__2Ue3h"]//p//text()')
                content = ' '.join(para)
                content = re.sub('  ', ' ', re.sub('\n|\r|\t|\xa0', ' ', content))
                f_content[a_idx]['content'] = content
                print(content)
                sleep(1.5)
        with open(content_path + str(file_idx) + '.txt', 'w') as f_check_w:
            f_check_w.write(ujson.dumps(f_content))
        file_idx += 1

    # clean data
    file_idx = 0
    total_cnt = 0
    while os.path.exists(content_path + str(file_idx) + '.txt'):

        with open(content_path + str(file_idx) + '.txt', 'r') as f_check:
            f_content = ujson.load(f_check)
        f_content_n = []
        for a_idx, a_dict in enumerate(f_content):
            if len(a_dict['content']) >= 10:
                f_content_n.append(a_dict)
        with open(content_path + str(file_idx) + '.txt', 'w') as f_check_w:
            f_check_w.write(ujson.dumps(f_content_n))
        print('Cleaning file index: {}, count: {}'.format(file_idx, len(f_content_n)))
        file_idx += 1
        total_cnt += len(f_content_n)
    print('Total news count: {}'.format(total_cnt))




# article_content_scraper(METADATA_FILE, ARTICLE_PATH, ARTICLE_BATCH_SIZE)
# article_content_check(ARTICLE_PATH)
# total_corpus = corpus_loader(ARTICLE_PATH)


