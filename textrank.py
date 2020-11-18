import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from bs4 import BeautifulSoup
import requests
import re
from rouge import Rouge


nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

vec_path = './glove.6B/glove.6B.100d.txt'
embeddings_file = open(vec_path, 'r')

embeddings = dict()

for line in embeddings_file:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float64')
    embeddings[word] = coefs
    

class Graph:
  
    def __init__(self,graph_dictionary):
        if not graph_dictionary:
            graph_dictionary={}
        self.graph_dictionary = graph_dictionary

    def vertices(self):
        return self.graph_dictionary.keys()

    def edges(self):
        return self.generate_edges()

    def add_vertex(self,vertex):
        if vertex not in graph_dictionary.keys():
            graph_dictionary[vertex] = []

    def add_edge(self,edge):
        vertex1,vertex2 = tuple(set(edge))
        if vertex1 in graph_dictionary.keys():
            graph_dictionary[vertex1].append(vertex2)
        else:
            graph_dictionary[vertex1] = [vertex2]

    def generate_edges(self):
        edges = set()
        for vertex in graph_dictionary.keys():
            for edges in graph_dictionary[vertex]:
                edges.add([vertex,edge])
        return list(edges)
    

def clean(sentence):
    lem = WordNetLemmatizer()
    sentence = sentence.lower()
    sentence = re.sub(r'http\S+',' ',sentence)
    sentence = re.sub(r'[^a-zA-Z]',' ',sentence)
    sentence = sentence.split()
    sentence = [lem.lemmatize(word) for word in sentence if word not in stopwords.words('english')]
    sentence = ' '.join(sentence)
    return sentence

def average_vector(sentence):
    words = sentence.split()
    size = len(words)
    average_vector = np.zeros((size,100))
    unknown_words=[]

    for index, word in enumerate(words):
        try:
            average_vector[index] = embeddings[word].reshape(1,-1)
        except Exception as e:
            unknown_words.append(word)
            average_vector[index] = 0

    if size != 0:
        average_vector = sum(average_vector)/size
    return average_vector,unknown_words

def cosine_similarity(s1, s2):
    v1, _ = average_vector(s1)
    v2, _ = average_vector(s2)
    cos_sim = 0
    try:
        cos_sim = (np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    except Exception as e :
        pass
    return cos_sim

def page_rank(graph, len_clean_sent, iterations = 50,sentences=20):
    ranks = []
    network = graph.graph_dictionary
    current_ranks = np.squeeze(np.zeros((1, len_clean_sent)))
    prev_ranks = np.array([1/len_clean_sent]*len_clean_sent)
    for iteration in range(0,iterations):
        for i in range(0,len(list(network.keys()))):
            current_score = 0
            adjacent_vertices = network[list(network.keys())[i]]
            for vertex in adjacent_vertices:
                current_score += prev_ranks[vertex]/len(network[vertex])
            current_ranks[i] = current_score
    prev_ranks = current_ranks

    for index in range(len_clean_sent):
        if prev_ranks[index]: 
            ranks.append((index,prev_ranks[index]))
    ranks = sorted(ranks,key = lambda x:x[1],reverse=True)[:sentences]

    return ranks

def textrank_summarise(paragraph, no_of_sentences):
    
    sentences = sent_tokenize(paragraph)
    cleaned_sentences=[]
    for sentence in sentences:
        cleaned_sentences.append(clean(sentence))
    len_clean_sent = len(cleaned_sentences)
    similarity_matrix = np.zeros((len_clean_sent,len_clean_sent))

    for i in range(0,len_clean_sent):
        for j in range(0,len_clean_sent):
            if type(cosine_similarity(cleaned_sentences[i],cleaned_sentences[j])) == np.float64 :
                similarity_matrix[i,j] = cosine_similarity(cleaned_sentences[i],cleaned_sentences[j])
                
    similarity_threshold = 0.70
    network_dictionary = {}

    for i in range(len_clean_sent):
        network_dictionary[i] = []  

    for i in range(len_clean_sent):
        for j in range(len_clean_sent):
            if similarity_matrix[i][j] > similarity_threshold:
                if j not in network_dictionary[i]:
                    network_dictionary[i].append(j)
                if i not in network_dictionary[j]:
                    network_dictionary[j].append(i)
    
    graph = Graph(network_dictionary)
    
    ranks = page_rank(graph, len_clean_sent, iterations=500, sentences=no_of_sentences)
    
    summary = ""
    ranks_sum = []
    for index,rank in ranks:  
        summary += sentences[index] + " "
        ranks_sum.append(rank)
    
    summary = summary.strip()
    
    return summary, ranks_sum

def bleu_score(original, generated):
    smoothing = SmoothingFunction().method0
    original_tokens = word_tokenize(original)
    generated_tokens = word_tokenize(generated)
    result = sentence_bleu(original_tokens, generated_tokens, smoothing_function=smoothing)
    return result

def rouge(original, generated):
    rouge = Rouge()
    score = rouge.get_scores(original,generated)
    rouge1f = score[0]['rouge-1']['f']
    rouge1p = score[0]['rouge-1']['p']
    rouge1r = score[0]['rouge-1']['r']
    rouge2f = score[0]['rouge-2']['f']
    rouge2p = score[0]['rouge-2']['p']
    rouge2r = score[0]['rouge-2']['r']
    rougelf = score[0]['rouge-l']['f']
    rougelp = score[0]['rouge-l']['p']
    rougelr = score[0]['rouge-l']['r']
    
    return (rouge1f, rouge1p, rouge1r), (rouge2f, rouge2p, rouge2r), (rougelf, rougelp, rougelr)

