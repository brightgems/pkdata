# coding: utf-8
import requests
import lxml.etree as et
from lxml.html.clean import clean_html
from tqdm import tqdm

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
        text += _pp[0].xpath("string(.)") +"\n"
    with open('corpus/hdnj.txt','w',encoding='utf-8') as f:
        f.write(text)

if __name__ == '__main__':
    download_hdnj()