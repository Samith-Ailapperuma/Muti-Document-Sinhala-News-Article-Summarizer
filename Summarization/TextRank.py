from sinling import SinhalaTokenizer, SinhalaStemmer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

sinhala_tokenizer = SinhalaTokenizer()
sinhala_stemmer = SinhalaStemmer()

# Function to tokenize and stem the sentences
def preprocess_text(text):
    words = sinhala_tokenizer.tokenize(text)
    words = [sinhala_stemmer.stem(word) for word in words]
    return words

# Function to calculate the cosine similarity between two sentences
def calculate_similarity(sentence1, sentence2):
    words1 = preprocess_text(sentence1)
    words2 = preprocess_text(sentence2)
    all_words = list(set(words1 + words2))

    vector1 = [1 if word in words1 else 0 for word in all_words]
    vector2 = [1 if word in words2 else 0 for word in all_words]

    return cosine_similarity([vector1], [vector2])[0][0]

# Function to build the TextRank graph
def build_graph(texts):
    G = nx.Graph()
    for i, text in enumerate(texts):
        sentences = sinhala_tokenizer.split_sentences(text)
        for sentence in sentences:
            G.add_node(f"{i}_{sentence}")
        for node1 in G.nodes():
            for node2 in G.nodes():
                if node1 != node2:
                    similarity = calculate_similarity(node1.split('_')[1], node2.split('_')[1])
                    G.add_edge(node1, node2, weight=similarity)
    return G

# Function to give aggregate score for each sentence 
def sentences_with_scores(texts):
    scored_sentences = {}
    G = build_graph(texts)
    scores = nx.pagerank(G)
    for key in scores:
        value = scores[key]
        modified_key = key.split('_')[1]
        scored_sentences[modified_key] = value
    return scored_sentences