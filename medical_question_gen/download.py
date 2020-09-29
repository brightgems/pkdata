# coding: utf-8
import requests
import lxml.etree as et
from lxml.html.clean import clean_html
from tqdm import tqdm
import re
import time
from splinter.browser import _DRIVERS
import requests
from splinter import Browser
from selenium.webdriver.common.proxy import Proxy, ProxyType
from concurrent.futures import ProcessPoolExecutor
from func_timeout import func_timeout,func_set_timeout, FunctionTimedOut
import logging
from retry import retry

def download_hdnj():
    """下载黄帝内经语料"""
    resp = requests.get("http://ewenyan.com/contents/more/hdnj.html")
    parser = et.HTMLParser()
    html = resp.content.decode('gb2312',errors="ignore")
    root =et.fromstring(html,parser)
    links=root.xpath('//tr/td/a')
    urls=[]
    for each in links:
        href=each.attrib['href']
        urls.append("http://ewenyan.com/"+href[6:])
    # download text
    text=""
    for url in tqdm(urls):
        if 'articles' not in url:
            continue
        resp = requests.get(url)
        html = resp.content.decode('gb2312',errors="ignore")
        dom =et.fromstring(html,parser)
        _pp=dom.cssselect("td>p")
        pt=_pp[0].xpath("string(.)")
        pt=re.sub('\r\n\s+','\n',pt).replace('\n\n','\n')
        text += pt +"\n"
    with open('corpus/hdnj.txt','w',encoding='utf-8') as f:
        f.write(text)

def download_bkmy():
    def init_home_page(home_url,proxy=None):
        # 使用selenium打开主页
        browser = Browser('chrome', headless=True)
        browser.visit(home_url)
        return browser
    browser = init_home_page('https://www.baikemy.com/disease/list/0/0?diseaseContentType=I')
    browser.execute_script("$(\"a[href='javascript:void(0);']\").click()")
    parser = et.HTMLParser()
    text=""
    time.sleep(0.5)
    for eachLi in tqdm(browser.find_by_css(".typeInfo_Li a")):
        if '更多' in eachLi.outer_html:
            continue
        li = et.fromstring(eachLi.outer_html)
        url = "https://www.baikemy.com"+li.attrib["href"]
        resp=requests.get(url)
        root =et.fromstring(resp.text,parser)
        for content in root.cssselect(".content"):
            pt = content.xpath("string(.)").strip()
            if pt:
                break
        pt=re.sub('\n\s+','\n',pt)
        text += eachLi.text+"\n"+ pt +"\n\n"
    with open('corpus/bkmy.txt','w',encoding='utf-8') as f:
        f.write(text)

if __name__ == '__main__':
    download_bkmy()