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
import textrank as tr

    
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


st.title('Text Summarization using TextRank Algorithm')
st.subheader('One stop for all types of summarizations!')


image = Image.open('circle-cropped-1.png')
st.sidebar.image(image, use_column_width=True)
st.sidebar.markdown('<center> <h1>A-1 Summit</h1></center>',unsafe_allow_html=True)
st.sidebar.markdown('[![]\
					(https://img.shields.io/badge/GitHub_Link-gray.svg?colorA=gray&colorB=gray&logo=github)]\
					(https://github.com/arghyadeep99/A1-Summit)')

st.sidebar.markdown('<center>The All-in-1 Summariser for your news articles, Wiki articles, notes and YouTube videos!</center>',unsafe_allow_html=True)
st.sidebar.markdown('<center> <h3>Made By <a href="https://github.com/RusherRG" target="_blank">Rushang Gajjal</a>, <a href="https://arghyadeepdas.tech" target="_blank">Arghyadeep Das</a> and <a href="https://kiteretsu.tech" target="_blank">Karan Sheth</a>.</h3></center>',unsafe_allow_html=True)

def print_usage():
    # Display the parameters and what they mean.
    st.write('''
    Usage:
        summarize.py <wiki-url> <summary length>
    Explanation:
        Parameter 1: Wikipedia URL to pull
        Parameter 2: the number of words for the summary to contain
    ''')

def summarize(article_text, num_of_sentences):
    # Extract keywords
    stop_words = set(stopwords.words('english')) 
    keywords = mz_keywords(article_text,scores=True,threshold=0.003)
    keywords_names = []
    for tuples in keywords:
        if tuples[0] not in stop_words: 
            if len(tuples[0]) > 2:
                keywords_names.append(tuples[0])

    
    pre_summary, rank_sum = tr.textrank_summarise(article_text, num_of_sentences)
    
    summary = re.sub("[\(\[].*?[\)\]♪]", "", pre_summary)
    
    print_pretty(summary, keywords_names)
    
    return summary, rank_sum

def print_pretty (summary, keywords_names):
    columns = os.get_terminal_size().columns
    
    printable = summary
    st.write(printable.center(columns))
    str_keywords_names = str(keywords_names).strip('[]')
    printable2 = str_keywords_names
    st.write("Keywords: ",printable2.center(columns))
    
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

def print_rouge(par, summ):
    (f11, p1, r1), (f12, p2, r2), (f1l, pl, rl)  = tr.rouge(par, summ)
    precision = [p1, p2, pl]
    recall = [r1, r2, rl]
    f1 = [f11, f12, f1l]
    df = pd.DataFrame(list(zip(precision, recall, f1)), columns=["Precision", "Recall", "F-1 Score"], index=["ROUGE-1", "ROUGE-2", "ROUGE-L"])
    st.subheader("ROUGE Scores")
    st.table(df)

url = st.text_input('\nEnter URL of news article from The Hindu Newspaper: ')

wikiurl = st.text_input('\nEnter URL of Wikipedia article: ')
video_id = st.text_input("\nEnter the Youtube Video Id:")
# video_id = "Na8vHaCLwKc" 
textfield123 = st.text_area('\nEnter article or paragraph you want to summarize ')
no_of_sentences = st.number_input('Choose the no. of sentences in the summary:', min_value = 1)

def textfunc():

    content = textfield123
    content = sanitize_input(content)

    text = re.sub(r'\[[0-9]*\]', ' ', content)
    text = re.sub(r'\s+', ' ', text)
    
    st.subheader('Original text: ')
    st.write(text)
    
    tokens = word_tokenize(text)
    sents = sent_tokenize(text)
    word_counts = count_words(tokens)
    freq_dist = word_freq_distribution(word_counts)
    sent_scores = score_sentences(sents, freq_dist)
    st.subheader('Summarised text: ')

    summary, sum_scores = summarize(text, no_of_sentences)
        
    subh = 'Summary sentence score for the top ' + str(no_of_sentences) + ' sentences: '

    st.subheader(subh)
    summ_sentences = sent_tokenize(summary)
    data = list(zip(summ_sentences, sum_scores))
        
    df = pd.DataFrame(data, columns = ['Sentence', 'Score'])

    st.table(df)
    print_rouge(text, summary)

def textforYT():

    transcript = ytapi.get_transcript(video_id)
    text = ""
    for i in transcript:
        text += i["text"]
        text += " "
    text.replace("\n"," ")

    sent_tokens, word_tokens = sent_tokenize(text), word_tokenize(text)
    sent_ranks = score_tokens(sent_tokens, word_tokens)    
    
    st.subheader('Summarised text: ')
    
    summary, sum_scores = summarize(text, no_of_sentences)

    
    subh = 'Summary sentence score for the top ' + str(no_of_sentences) + ' sentences: '

    st.subheader(subh)
    summ_sentences = sent_tokenize(summary)
    data = list(zip(summ_sentences, sum_scores))
        
    df = pd.DataFrame(data, columns = ['Sentence', 'Score'])

    st.table(df)
    print_rouge(text, summary)


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
    
    tokens = word_tokenize(text)
    sents = sent_tokenize(text)
    word_counts = count_words(tokens)
    freq_dist = word_freq_distribution(word_counts)
    sent_scores = score_sentences(sents, freq_dist)
    summary, sum_scores = summarize(text, no_of_sentences)
    
    st.subheader('Summarised text: ')
    st.write(summary)
    
    subh = 'Summary sentence score for the top ' + str(no_of_sentences) + ' sentences: '

    st.subheader(subh)
    summ_sentences = sent_tokenize(summary)
    data = list(zip(summ_sentences, sum_scores))
        
    df = pd.DataFrame(data, columns = ['Sentence', 'Score'])

    st.table(df)
    print_rouge(text, summary)

if wikiurl and no_of_sentences and st.button('Summarize Wikipedia Article'):
    if not str(no_of_sentences).isdigit():
        print_usage()
    else:
        scraped_data = urllib.request.urlopen(wikiurl)  
        article = scraped_data.read()

        parsed_article = bs.BeautifulSoup(article,'lxml')
        paragraphs = parsed_article.find_all('p')
        article_text = ""
        for p in paragraphs:  
            article_text += p.text
        
        
        st.subheader('Summarised text: ')
        summary, sum_scores = summarize(article_text, int(no_of_sentences))

        subh = 'Summary sentence score for the top ' + str(no_of_sentences) + ' sentences: '

        st.subheader(subh)
        summ_sentences = sent_tokenize(summary)
        data = list(zip(summ_sentences, sum_scores))

        df = pd.DataFrame(data, columns = ['Sentence', 'Score'])

        st.table(df)
        print_rouge(article_text, summary)

if textfield123 and no_of_sentences and st.button('Summarize Text'):
    if not str(no_of_sentences).isdigit():
        st.write("Use it again. Error occured summarizing article.")
    else:
        textfunc()
if video_id and no_of_sentences and st.button('Summarize YouTube video'):
    if not str(no_of_sentences).isdigit():
        st.write("Use it again. Error occured summarizing article.")
    else:
        textforYT()

