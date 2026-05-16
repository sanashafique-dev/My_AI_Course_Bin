'''import requests
from bs4 import BeautifulSoup
import csv

URL='https://www.alibaba.com/'
r = requests.get(URL)
 
soup = BeautifulSoup(r.content, 'html5lib')
quotes=[]  
 
table = soup.find('div', attrs = {'id':'all_items'}) 
'''
import requests
from bs4 import BeautifulSoup

URL='https://quotes.toscrape.com/'
r=requests.get(URL)

soup=BeautifulSoup(r.text,'html5lib')

quotes=soup.find_all('span',class_='text')
authors=soup.find_all('small',class_='author')

for row  in range(len(quotes)):
     
    print("Quote:", quotes[row].text)
    print("Author:", authors[row].text)
    


  


 
