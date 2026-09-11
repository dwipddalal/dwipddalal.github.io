#!/usr/bin/env python3
"""Check CityNav's citation identity and crawlable assets using only Python stdlib.

Run from any directory: python3 scripts/check_citynav_seo.py
This validates the files; it does not measure rankings or request reindexing.
"""

from collections import defaultdict
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlparse
import xml.etree.ElementTree as ET


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.meta = defaultdict(list)
        self.links = []
        self.ids = set()
        self.feed(source)
        self.entities = []
        for raw in re.findall(r'<script type="application/ld\+json">(.*?)</script>', source, re.S):
            data = json.loads(raw)
            self.entities.extend(data.get('@graph', [data]))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta':
            self.meta[attrs.get('name', attrs.get('property'))].append(attrs.get('content'))
        if tag == 'link':
            self.links.append(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])


def plain(text):
    return re.sub(r'\s+', ' ', unescape(re.sub(r'<[^>]*>', '', text))).strip()


root = Path(__file__).resolve().parents[1]
source = (root / 'AgentNav/index.html').read_text()
page = Page(source)
home = Page((root / 'index.html').read_text())
canonical = 'https://dwipddalal.github.io/AgentNav/'
doi = '10.18653/v1/2026.eacl-long.387'
article_id = canonical + '#article'
articles = [n for n in page.entities if n.get('@type') == 'ScholarlyArticle']
assert len(articles) == 1, 'Use one preferred paper entity for both titles.'
article = articles[0]
assert article['@id'] == article_id
assert 'City Navigation in the Wild' in article['alternateName']
assert page.meta['citation_title'] == [article['name']]
assert plain(re.search(r'<h1\b[^>]*>(.*?)</h1>', source, re.S).group(1)) == article['name']
assert page.meta['citation_author'] == ['Dalal, Dwip', 'Mishra, Utkarsh', 'Ahuja, Narendra', 'Jojic, Nebojsa']
assert page.meta['citation_publication_date'] == ['2026']
assert page.meta['citation_doi'] == [doi]
assert page.meta['citation_arxiv_id'] == ['2512.15933']
assert page.meta['citation_firstpage'] == ['8279']
assert page.meta['citation_lastpage'] == ['8303']
assert page.meta['citation_abstract_html_url'] == [canonical]
assert [x['href'] for x in page.links if x.get('rel') == 'canonical'] == [canonical]
assert [x['href'] for x in home.links if x.get('rel') == 'canonical'] == ['https://dwipddalal.github.io/']
assert not any('noindex' in value.lower() for value in page.meta['robots'])
home_article = next(n for n in home.entities if n.get('@id') == article_id)
for field in ['name', 'alternateName', 'author', 'identifier', 'sameAs']:
    assert home_article[field] == article[field], f'Homepage differs: {field}'
assert plain(article['abstract']) in plain(source.split('<body>', 1)[1])
assert 'citynav' in page.ids
assert any(n.get('@type') == 'Dataset' and n.get('name') == 'CityNav' for n in page.entities)
assert any(n.get('@type') == 'SoftwareSourceCode' and n.get('name') == 'AgentNav' for n in page.entities)

pdf_url, = page.meta['citation_pdf_url']
assert pdf_url.rsplit('/', 1)[0] + '/' == canonical, 'Scholar PDF must share the abstract page directory.'
pdf = root / urlparse(pdf_url).path.lstrip('/')
assert pdf.suffix == '.pdf' and pdf.read_bytes().startswith(b'%PDF-')
assert pdf.stat().st_size < 5_000_000, 'Google Scholar requires files below 5 MB.'
assert article['encoding']['contentUrl'] == pdf_url
urls = [el.text for el in ET.parse(root / 'sitemap.xml').findall('.//{*}loc')]
assert len(urls) == len(set(urls))
assert canonical in urls and pdf_url in urls

bib = (root / 'AgentNav/citation.bib').read_text().strip()
inline = unescape(re.search(r'<pre><code>(@inproceedings\{.*?)</code></pre>', source, re.S).group(1)).strip()
assert bib == inline, 'Downloadable and displayed BibTeX must agree.'
assert doi in bib and '8279--8303' in bib
ris = (root / 'AgentNav/citation.ris').read_text()
assert 'DO  - ' + doi in ris and 'TI  - ' + article['name'] in ris
for file in ['llms.txt', 'AgentNav/llms.txt']:
    text = (root / file).read_text()
    assert article['name'] in text and article['alternateName'] in text and doi in text
print(f'PASS: paper identity, structured data, citations, sitemap, and {pdf.stat().st_size:,}-byte PDF.')
