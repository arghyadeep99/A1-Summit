import streamlit as st
from bs4 import BeautifulSoup
import requests
import re
from collections import Counter 
from string import punctuation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
from youtube_transcript_api import YouTubeTranscriptApi as ytapi
import pandas as pd
import bs4 as bs  
import urllib.request  
import re
from PIL import Image
from gensim.summarization import summarize as su_gs
from gensim.summarization import keywords
from gensim.summarization import mz_keywords
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from sys import argv
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict

    
def tokenizer(s):
    tokens = []
    for word in s.split(' '):
        tokens.append(word.strip().lower())
        
    return tokens

def sent_tokenizer(s):
    sents = []
    for sent in s.split('.'):
        sents.append(sent.strip())
        
    return sents

def count_words(tokens):
    word_counts = {}
    for token in tokens:
        if token not in stop_words and token not in punctuation:
            if token not in word_counts.keys():
                word_counts[token] = 1
            else:
                word_counts[token] += 1
                
    return word_counts

def word_freq_distribution(word_counts):
    freq_dist = {}
    max_freq = max(word_counts.values())
    for word in word_counts.keys():  
        freq_dist[word] = (word_counts[word]/max_freq)
        
    return freq_dist

def score_sentences(sents, freq_dist, max_len=40):
    sent_scores = {}  
    for sent in sents:
        words = sent.split(' ')
        for word in words:
            if word.lower() in freq_dist.keys():
                if len(words) < max_len:
                    if sent not in sent_scores.keys():
                        sent_scores[sent] = freq_dist[word.lower()]
                    else:
                        sent_scores[sent] += freq_dist[word.lower()]
                        
    return sent_scores

def summarize(sent_scores, k):
    top_sents = Counter(sent_scores) 
    summary = ''
    scores = []
    
    top = top_sents.most_common(k)
    
    for t in top: 
        summary += t[0].strip() + '. '
        scores.append((t[1], t[0]))
        
    return summary[:-1], scores


st.title('Text Summarization')
st.subheader('One stop for all types of summarizations')


image = Image.open('circle-cropped-1.png')
st.sidebar.image(image, use_column_width=True)
st.sidebar.markdown('<center> <h1>A-1 Summit</h1></center>',unsafe_allow_html=True)
st.sidebar.markdown('[![]\
					(https://img.shields.io/badge/GitHub_Link-gray.svg?colorA=gray&colorB=gray&logo=github)]\
					(https://github.com/arghyadeep99/A1-Summit)')

st.sidebar.markdown('<center>The All-in-1 Summariser for your news articles, Wiki articles, notes and YouTube videos!</center>',unsafe_allow_html=True)
st.sidebar.markdown('<center> <h3>Made By <a href="https://github.com/RusherRG" target="_blank">Rushang Gajjal</a>, <a href="https://arghyadeepdas.tech" target="_blank">Arghyadeep Das</a> and <a href="https://kiteretsu.tech" target="_blank">Karan Sheth</a>.</h3></center>',unsafe_allow_html=True)




url = st.text_input('\nEnter URL of news article from The Hindu Newspaper: ')

wikiurl = st.text_input('\nEnter URL of Wikipedia article: ')
video_id = st.text_input("\nEnter the Youtube Video Id:")
# video_id = "Na8vHaCLwKc" 
textfield123 = st.text_area('\nEnter article or paragraph you want to summarize ')
no_of_sentences = st.number_input('Choose the no. of sentences in the summary (no. of words for Wiki article)', min_value = 1)

def textfunc():

    content = textfield123
    content = sanitize_input(content)

    sent_tokens, word_tokens = tokenize_content(content)
    sent_ranks = score_tokens(sent_tokens, word_tokens)
    st.write(summarize2(sent_ranks, sent_tokens, no_of_sentences))

def textforYT():

    a = ytapi.get_transcript(video_id)
    textstr = ""
    for i in a:
        textstr += i["text"]
    article = [textstr[i:i + 100] for i in range(0, len(textstr), 100)]
    res = '. '.join(textstr[i:i + 100] for i in range(0, len(textstr), 100))
    content = res


    sent_tokens, word_tokens = tokenize_content(content)
    sent_ranks = score_tokens(sent_tokens, word_tokens)
    st.write(summarize2(sent_ranks, sent_tokens, no_of_sentences))


def tokenize_content(content):
    stop_words = set(stopwords.words('english') + list(punctuation))
    words = word_tokenize(content.lower())
    return (sent_tokenize(content), [word for word in words if word not in stop_words])

def score_tokens(sent_tokens, word_tokens):
    word_freq = FreqDist(word_tokens)
    rank = defaultdict(int)
    for i, sent in enumerate(sent_tokens):
        for  word in word_tokenize(sent.lower()):
            if word in word_freq:
                rank[i] += word_freq[word]
    return rank

def sanitize_input(data):
    replace = {
        ord('\f') : ' ',
        ord('\t') : ' ',
        ord('\n') : ' ',
        ord('\r') : None
    }
    return data.translate(replace)

def summarize2(ranks, sentences, length):

    if int(length) > len(sentences):
        print('You requested more sentences in the summary than there are in the text.')
        return ''

    else:
        indices = nlargest(int(length), ranks, key=ranks.get)
        final_summary = [sentences[j] for j in indices]
        return ' '.join(final_summary)



if url and no_of_sentences and st.button('Summarize The Hindu Article'):
    text = ""
    
    r=requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser') 
    content = soup.find('div', attrs = {'id' : re.compile('content-body-14269002-*')})
    
    for p in content.findChildren("p", recursive = 'False'):
        text+=p.text+" "
            
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    st.subheader('Original text: ')
    st.write(text)
    
    tokens = tokenizer(text)
    sents = sent_tokenizer(text)
    word_counts = count_words(tokens)
    freq_dist = word_freq_distribution(word_counts)
    sent_scores = score_sentences(sents, freq_dist)
    summary, summary_sent_scores = summarize(sent_scores, no_of_sentences)
    
    st.subheader('Summarised text: ')
    st.write(summary)
    
    subh = 'Summary sentence score for the top ' + str(no_of_sentences) + ' sentences: '

    st.subheader(subh)
    
    data = []

    for score in summary_sent_scores: 
        data.append([score[1], score[0]])
        
    df = pd.DataFrame(data, columns = ['Sentence', 'Score'])

    st.table(df)
    st.info('An application made by Piyush and Aditi')

def print_usage():
    # Display the parameters and what they mean.
    st.write('''
    Usage:
        summarize.py <wiki-url> <summary length>
    Explanation:
        Parameter 1: Wikipedia URL to pull
        Parameter 2: the number of words for the summary to contain
    ''')

def summarize(url_topull, num_of_words):
    # Obtain text
    scraped_data = urllib.request.urlopen(url_topull)  
    article = scraped_data.read()
    
    parsed_article = bs.BeautifulSoup(article,'html.parser')
    paragraphs = parsed_article.find_all('p')
    article_text = ""
    for p in paragraphs:  
        article_text += p.text

    # Extract keywords
    stop_words = set(stopwords.words('english')) 
    keywords = mz_keywords(article_text,scores=True,threshold=0.003)
    keywords_names = []
    for tuples in keywords:
        if tuples[0] not in stop_words: 
            if len(tuples[0]) > 2:
                keywords_names.append(tuples[0])

    
    pre_summary = su_gs(article_text,word_count=num_of_words)
    
    summary = re.sub("[\(\[].*?[\)\]]", "", pre_summary)
    
    print_pretty (summary,keywords_names)

def print_pretty (summary, keywords_names):
    columns = os.get_terminal_size().columns
    
    printable = summary
    st.write(printable.center(columns))
    str_keywords_names = str(keywords_names).strip('[]')
    printable2 = str_keywords_names
    st.write(printable2.center(columns))

if wikiurl and no_of_sentences and st.button('Summarize WikiPedia Article'):
    if not str(no_of_sentences).isdigit():
        print_usage()
    else:
        summarize(wikiurl, int(no_of_sentences))

if textfield123 and no_of_sentences and st.button('Summarize Text'):
    if not str(no_of_sentences).isdigit():
        st.write("Use it again. Error occured summarizing article.")
    else:
        textfunc()
if video_id and no_of_sentences and st.button('Summarize Youtube video'):
    if not str(no_of_sentences).isdigit():
        st.write("Use it again. Error occured summarizing article.")
    else:
        textforYT()
