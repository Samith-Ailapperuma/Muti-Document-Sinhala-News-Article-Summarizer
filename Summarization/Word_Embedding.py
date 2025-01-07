import fasttext
import numpy as np
from sinling import SinhalaTokenizer

sinhala_tokenizer = SinhalaTokenizer()

# Load pre-trained fastText model
model = fasttext.load_model(r'D:\Lecture Notes\Level 4 Semester 1\IS 4910 - Comprehensive Group Project\FastText\cc.si.300.bin')

# Function to generate embeddings for each summary
def generate_document_embeddings(documents):
    document_embeddings = []
    for document in documents:
        # Get embeddings for each word in the summary
        word_embeddings = [model.get_word_vector(word) for word in document.split()]
        # Average the word embeddings to get the document embedding
        if word_embeddings:
            document_embedding = np.mean(word_embeddings, axis=0)
            document_embeddings.append(document_embedding)
    if document_embeddings:
        document_embedding = np.mean(document_embeddings, axis=0)
        return document_embedding
    else:
        return None

# Function to get sentence embeddings 
def get_sentence_embeddings(sentences):
    sentence_embeddings = {}
    for sentence in sentences:
        # Get the word embedding for each word in a sentence
        word_embeddings = [model.get_word_vector(word) for word in sentence.split()]
        if word_embeddings:
            # Get sentence embedding by mean of word embeddings 
            sentence_embedding = np.mean(word_embeddings, axis=0)
            sentence_embeddings[sentence] = sentence_embedding
    return sentence_embeddings     

# Function to give similarity score between sentence and documents
def sentence_score(sentence_embeddings, document_embedding):
    similarity_scores = {}
    for key in sentence_embeddings:
        sent_embedding = sentence_embeddings[key]
        # Calculate euclidian distance
        euc_distance = np.linalg.norm(document_embedding - sent_embedding)
        similarity_scores[key] = euc_distance
    return similarity_scores