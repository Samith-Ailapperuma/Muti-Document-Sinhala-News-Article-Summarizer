import TextRank as tr
import Word_Embedding as we
from nltk import sent_tokenize
from rouge import Rouge
from sinling import SinhalaTokenizer

input_file_path = r"D:\Lecture Notes\Level 4 Semester 1\IS 4910 - Comprehensive Group Project\Module Input and Output\Input Documents\Documents.txt"
output_file_path = r"D:\Lecture Notes\Level 4 Semester 1\IS 4910 - Comprehensive Group Project\Module Input and Output\Output Document\Summary.txt"

sinhala_tokenizer = SinhalaTokenizer()

# Function to read documents from the input file
def read_documents(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        documents = file.read().split('End of document')
    return [doc.replace('\n', '').strip() for doc in documents if doc.strip()]

documents = read_documents(input_file_path)

# Function to return sentence list
def get_sentence_list(documents):
    sentences = []
    for i, document in enumerate(documents):
        sentence = sinhala_tokenizer.split_sentences(document)
        sentences.extend(sentence)
    return sentences 

# Function to remove duplicate sentences
def remove_duplicates(sentences):
    threshold = 0.5
    new_sentence_list = []
    for i in range(len(sentences)):
        flag = True
        for j in range(i + 1, len(sentences)):
            rouge_score = calculate_rouge_score(sentences[i], sentences[j])
            if rouge_score > threshold:
                flag = False
                break
        if(bool(flag)):
            new_sentence_list.append(sentences[i])
    return new_sentence_list

# Function to calculate ROUGE score between sentences
def calculate_rouge_score(sentence_1, sentence_2):
    rouge = Rouge()
    score = rouge.get_scores(sentence_1, sentence_2)
    return score[0]['rouge-1']['f']

# Function to return combined score
def combine_scores(textrank_scores, word_embedding_scores):
    combined_score_dict = {}
    for key in textrank_scores:
        textrank_val = textrank_scores[key]
        word_embed_val = word_embedding_scores[key] 
        combined_score = 0.4 * textrank_val + 0.6 * (1-word_embed_val)
        combined_score_dict[key] = combined_score
    return combined_score_dict

# Function to rank sentences
def rank_sentences(combined_scores):
    ranked_sentences = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_sentences

# Function to select required number of sentences for summary
def select_sentences(ranked_sentences, num_sentences=5):
    final_list = []
    selected_sentences = ranked_sentences[:num_sentences]
    for list_item in selected_sentences:
        sentence = [item for item in list_item if isinstance(item, str)]
        final_list.append(sentence)
    modified_list = [sentence[0] for sentence in final_list]
    return modified_list

# Function to write summary to output file
def write_summaries(file_path, summary):
    with open(file_path, 'w', encoding="utf8") as file:
        summary_string = ''.join(summary)
        file.write(summary_string)

# Tokenize the sentences and get a sentence list
sentence_list = get_sentence_list(documents)

# Remove duplicate sentences
sentence_list_duplicates_removed = remove_duplicates(sentence_list)

# Textrank scores for each sentence
textrank_scores = tr.sentences_with_scores(sentence_list_duplicates_removed)

# Document embedding by considering documents
document_embeddings = we.generate_document_embeddings(documents)

# Sentence embeddings by considering sentences
sentence_embeddings = we.get_sentence_embeddings(sentence_list_duplicates_removed)

# Similarity scores by combining the document and sentence embeddings
word_embedding_scores = we.sentence_score(sentence_embeddings, document_embeddings)

# Combined score by combining textrank score and similarity score
combined_scores = combine_scores(textrank_scores, word_embedding_scores)

# Rank sentences based on the combined score
ranked_sentences = rank_sentences(combined_scores)

# Extract number of sentences from ranked sentences
summary = select_sentences(ranked_sentences)

# Write summaries to the output file
write_summaries(output_file_path, summary)