# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:04:00 2020

@author: Sajid Hasan Sifat
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 06:36:01 2019

@author: Sajid Hasan Sifat
"""
# summarizer
# pip install beautifulsoup4
# python -m pip install lxml


import bs4 as bs
import urllib.request
import re
import nltk
nltk.download('stopwords')
import heapq
#paster your URL
source = urllib.request.urlopen('https://pnachforon.blogspot.com/').read()
soup = bs.BeautifulSoup(source, 'lxml')

# get text
text1 = ""
text2 = ""
text3 = ""
# text1 = soup.get_text()
# text2 = soup.get_text()
# text3 = soup.get_text()


# parse url data

for paragraph in soup.find_all('div'):
    text1 += paragraph.text

for paragraph in soup.find_all('p'):
    text2 += paragraph.text

for paragraph in soup.find_all('tr'):
    text3 += paragraph.text

text = ""
text = text1 + text2 + text3
# preprocessing data

text = re.sub(r'\[[0-9]*\]+', ' ', text)
text = re.sub(r'\s+', ' ', text)
clean_text = text.lower()
clean_text = re.sub(r'\W', ' ', clean_text)
clean_text = re.sub(r'\d', ' ', clean_text)
clean_text = re.sub(r'\s+', ' ', clean_text)
cleane_text = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]', r'\1', clean_text)


# tokenize
sentences = nltk.sent_tokenize(text)

stop_words = nltk.corpus.stopwords.words('english')
additional_stopwords =  "অবশ্য অনেক অনেকে অনেকেই অন্তত অথবা অথচ অর্থাত অন্য আজ আছে আপনার আপনি আবার আমরা আমাকে আমাদের আমার আমি আরও আর আগে আগেই আই অতএব আগামী অবধি অনুযায়ী আদ্যভাগে এই একই একে একটি এখন এখনও এখানে এখানেই এটি এটা এটাই এতটাই এবং একবার এবার এদের এঁদের এমন এমনকী এল এর এরা এঁরা এস এত এতে এসে একে এ ঐ  ই ইহা ইত্যাদি উনি উপর উপরে উচিত ও ওই ওর ওরা ওঁর ওঁরা ওকে ওদের ওঁদের ওখানে কত কবে করতে কয়েক কয়েকটি করবে করলেন করার কারও করা করি করিয়ে করার করাই করলে করলেন করিতে করিয়া করেছিলেন করছে করছেন করেছেন করেছে করেন করবেন করায় করে করেই কাছ কাছে কাজে কারণ কিছু কিছুই কিন্তু কিংবা কি কী কেউ কেউই কাউকে কেন কে কোনও কোনো কোন কখনও ক্ষেত্রে খুব গুলি গিয়ে গিয়েছে গেছে গেল গেলে গোটা চলে ছাড়া ছাড়াও ছিলেন ছিল জন্য জানা ঠিক তিনি তিনঐ তিনিও তখন তবে তবু তাঁদের তাঁাহারা তাঁরা তাঁর তাঁকে তাই তেমন তাকে তাহা তাহাতে তাহার তাদের তারপর তারা তারৈ তার তাহলে তিনি তা তাও তাতে তো তত তুমি তোমার তথা থাকে থাকা থাকায় থেকে থেকেও থাকবে থাকেন থাকবেন থেকেই দিকে দিতে দিয়ে দিয়েছে দিয়েছেন দিলেন দু দুটি দুটো দেয় দেওয়া দেওয়ার দেখা দেখে দেখতে দ্বারা ধরে ধরা নয় নানা না নাকি নাগাদ নিতে নিজে নিজেই নিজের নিজেদের নিয়ে নেওয়া নেওয়ার নেই নাই পক্ষে পর্যন্ত পাওয়া পারেন পারি পারে পরে পরেই পরেও পর পেয়ে প্রতি প্রভৃতি প্রায় ফের ফলে ফিরে ব্যবহার বলতে বললেন বলেছেন বলল বলা বলেন বলে বহু বসে বার বা বিনা বরং বদলে বাদে বার বিশেষ বিভিন্ন বিষয়টি ব্যবহার ব্যাপারে ভাবে ভাবেই মধ্যে মধ্যেই মধ্যেও মধ্যভাগে মাধ্যমে মাত্র মতো মতোই মোটেই যখন যদি যদিও যাবে যায় যাকে যাওয়া যাওয়ার যত যতটা যা যার যারা যাঁর যাঁরা যাদের যান যাচ্ছে যেতে যাতে যেন যেমন যেখানে যিনি যে রেখে রাখা রয়েছে রকম শুধু সঙ্গে সঙ্গেও সমস্ত সব সবার সহ সুতরাং সহিত সেই সেটা সেটি সেটাই সেটাও সম্প্রতি সেখান সেখানে সে স্পষ্ট স্বয়ং হইতে হইবে হৈলে হইয়া হচ্ছে হত হতে হতেই হবে হবেন হয়েছিল হয়েছে হয়েছেন হয়ে হয়নি হয় হয়েই হয়তো হল হলে হলেই হলেও হলো হিসাবে হওয়া হওয়ার হওয়ায় হন হোক জন জনকে জনের জানতে জানায় জানিয়ে জানানো জানিয়েছে জন্য জন্যওজে জে বেশ দেন তুলে ছিলেন চান চায় চেয়ে মোট যথেষ্ট টি"
stop_words += additional_stopwords.split()

# histogram

word2count = {}
for word in nltk.word_tokenize(clean_text):
    if word not in stop_words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1

# weighted histogram
for key in word2count.keys():
    word2count[key] = word2count[key] / max(word2count.values())

# scoring sentences
sent2score = {}
for sentence in sentences:
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word2count.keys():
            if len(sentence.split(' ')) < 35 and len(sentence.split(' ')) > 8:
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]

# summarizing

best_sentences = heapq.nlargest(5, sent2score, key=sent2score.get)

print('------------- Summary --------------')
for sentences in best_sentences:
    print(sentences)

