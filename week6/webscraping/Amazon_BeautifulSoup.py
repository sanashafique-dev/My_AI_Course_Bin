import requests
from bs4 import BeautifulSoup
import csv

url='https://www.amazon.com/gp/browse.html?node=6563140011&ref_=nav_em_amazon_smart_home_0_2_8_2'
headers = {
    "User-Agent": "Mozilla/5.0"
}
r= requests.get(url,headers=headers)
soup=BeautifulSoup(r.content,'html5lib')
productlist=[]
products=soup.find_all('li',attrs={'class':"a-carousel-card ucw-widget-carousel-element"})

for item in products:
    product={}
    title=item.find('div',class_='ucw-widget-product-card-title').text
    rating=item.find('div',class_='ucw-widget-product-card-review').text
    price=item.find('div',class_='ucw-widget-product-card-price').text
    shiping=item.find('div',class_='ucw-widget-product-card-delivery').text


    product['title']=title
    product['rating']=rating
    product['price']=price
    product['shiping']=shiping

    productlist.append(product)

filename="week6/Amazon_beautifulSoup.csv"
with open(filename,'w',newline="",)as f:
    w=csv.DictWriter(f,['title','rating','price','shiping'])
    w.writeheader()
    for product in productlist:
        w.writerow(product)
