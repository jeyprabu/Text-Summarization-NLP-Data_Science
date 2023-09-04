import nltk
import numpy as np
from nltk.corpus import stopwords, state_union
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tag import StanfordNERTagger
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


stanford_ner_path = 'E:\\internship\\nep stanford\\stanford-ner-2020-11-17\\stanford-ner.jar'
stanford_classifier_path = 'E:\\internship\\nep stanford\\stanford-ner-2020-11-17\\classifiers\\english.muc.7class.distsim.crf.ser.gz'
ner_tagger = StanfordNERTagger(stanford_classifier_path, stanford_ner_path)

def preprocess_text(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return sentences, words

def sentence_scores_freq(sentences, words):
    freq_dist = FreqDist(words)
    sentence_scores = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in freq_dist.keys():
                if sentence not in sentence_scores.keys():
                    sentence_scores[sentence] = freq_dist[word]
                else:
                    sentence_scores[sentence] += freq_dist[word]
    return sentence_scores

def sentence_scores_tfidf(sentences, words):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    cosine_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_scores[sentence] = sum(cosine_matrix[i])
    
    return sentence_scores

def sentence_scores_cosine(sentences, words):
    def sentence_vector(sentence, words):
        sentence = ' '.join(words)
        vector = np.zeros(len(words))
        for word in words:
            vector += cosine_distance(
                nltk.FreqDist(nltk.word_tokenize(sentence.lower())),
                nltk.FreqDist(nltk.word_tokenize(word.lower()))
            )
        return vector

    sentence_vectors = [sentence_vector(sentence, words) for sentence in sentences]
    similarity_matrix = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = 1 - cosine_distance(sentence_vectors[i], sentence_vectors[j])

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_scores[sentence] = sum(similarity_matrix[i])
    
    return sentence_scores

def sentence_position_weights(sentences):
    num_sentences = len(sentences)
    position_scores = {sentences[i]: (1 - i / num_sentences) for i in range(num_sentences)}
    return position_scores

def include_proper_nouns(sentence_scores, sentences):
    tagged_sentences = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in sentences]
    proper_nouns_scores = {}

    for i, tagged_sentence in enumerate(tagged_sentences):
        for word, pos in tagged_sentence:
            if pos == 'NNP':  
                proper_nouns_scores[sentences[i]] = sentence_scores.get(sentences[i], 0) + 1

    return proper_nouns_scores

def normalize_sentence_scores(sentence_scores, sentences):
    max_score = max(sentence_scores.values())
    normalized_scores = {sentence: score / max_score for sentence, score in sentence_scores.items()}
    return normalized_scores

def extract_named_entities(text):
    named_entities = set()
    for sentence in sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence))):
            if hasattr(chunk, 'label') and chunk.label() in ['ORGANIZATION', 'PERSON', 'LOCATION']:
                named_entities.add(' '.join(c[0] for c in chunk))
    return list(named_entities)

def extract_keywords_tfidf(sentences):
    vectorizer = TfidfVectorizer(max_features=10)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).tolist()[0]
    keywords = [feature_names[i] for i in np.argsort(tfidf_scores)[::-1]]
    return keywords

def perform_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

def cluster_sentences(sentences, num_clusters=3):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    cluster_labels = kmeans.labels_
    return cluster_labels

def perform_pca(X, num_components=2):
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(X)
    return pca_result

def generate_summary(input_text, method='frequency', ratio=0.2):
    sentences, words = preprocess_text(input_text)
    
    if method == 'frequency':
        sentence_scores = sentence_scores_freq(sentences, words)
    elif method == 'tfidf':
        sentence_scores = sentence_scores_tfidf(sentences, words)
    elif method == 'cosine':
        sentence_scores = sentence_scores_cosine(sentences, words)
    else:
        raise ValueError("Invalid summarization method. Choose 'frequency', 'tfidf', or 'cosine'.")

    position_weights = sentence_position_weights(sentences)
    proper_nouns_scores = include_proper_nouns(sentence_scores, sentences)
    normalized_scores = normalize_sentence_scores(sentence_scores, sentences)

    final_scores = {}
    for sentence in sentences:
        final_scores[sentence] = (
            sentence_scores.get(sentence, 0) +
            position_weights.get(sentence, 0) +
            proper_nouns_scores.get(sentence, 0) +
            normalized_scores.get(sentence, 0)
        )


    
    sorted_sentences = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    num_sentences = int(len(sentences) * ratio)
    top_sentences = sorted_sentences[:num_sentences]
    top_sentences = [s[0] for s in top_sentences]
    summary = ' '.join(top_sentences)
    
    return summary



