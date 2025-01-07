import fasttext
import numpy as np
# from ..Summarization.Combine import documents, summary

# Load pre-trained fastText model
model = fasttext.load_model(
    # File path
)

summary_file_path = ""
document_file_path = ""

# Function to read documents from the input file
def read_summary(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        summary = file.read()
        # documents = file.read().split('End of document')
    # return [doc.replace('\n', '').strip() for doc in documents if doc.strip()]
    return summary

summary = read_summary(summary_file_path)

def read_documents(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        documents = file.read().split('End of document')
    return [doc.replace('\n', '').strip() for doc in documents if doc.strip()]

documents = read_documents(document_file_path)

def get_summary_embedding(summary):
    word_embeddings = [model.get_word_vector(word) for word in summary.split()]
    if word_embeddings:
        summary_embedding = np.mean(word_embeddings, axis=0)
    return summary_embedding

def get_document_embeddings(documents):
    document_embeddings = []
    for document in documents:
        # Get embeddings for each word in the summary
        word_embeddings = [model.get_word_vector(word) for word in document.split()]
        # Average the word embeddings to get the document embedding
        if word_embeddings:
            document_embedding = np.mean(word_embeddings, axis=0)
            document_embeddings.append(document_embedding)
    return document_embeddings

def compare_similarity(summary_embedding, document_embeddings):
    similarity_list = []
    for document_embedding in document_embeddings:
        euc_distance = np.linalg.norm(document_embedding - summary_embedding)
        similarity_list.append(euc_distance)
    return similarity_list

def describe_similarity(similarity_list):
    for item in similarity_list:
        result = f"The euclidian distance between the summary and the {similarity_list.index(item)} th document is {item}"
        print(result)

# documents = cmb.documents
# summary = cmb.summary

summary_embedding = get_summary_embedding(summary)

document_embeddings = get_document_embeddings(documents)

similarity_list = compare_similarity(summary_embedding, document_embeddings)

describe_similarity(similarity_list)