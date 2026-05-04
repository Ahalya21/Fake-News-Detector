import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



#Load saved models
tfidf    = joblib.load('tfidf_vectorizer.pkl')
scaler   = joblib.load('nb_scaler.pkl')
lr_model = joblib.load('lr_model.pkl')
nb_model = joblib.load('nb_model.pkl')


#1 Load held-out test set
df_held = pd.read_csv('held_out_test.csv')

print(f"Held-out shape : {df_held.shape}")
print(f"Label distribution:\n{df_held['label'].value_counts()}")


#2 preprocessing

print('------------Preprocessing------------')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#2.1 lowercase
df_held['content'] = df_held['content'].str.lower()

#2.2 remove noise
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_held['content'] = df_held['content'].apply(clean_text)

#2.3 remove stopwords
def remove_stopwords(text):
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df_held['content'] = df_held['content'].apply(remove_stopwords)

#2.4 lemmatization
def lemmatize_text(text):
    tokens = text.split()
    return ' '.join([lemmatizer.lemmatize(w) for w in tokens])

df_held['content'] = df_held['content'].apply(lemmatize_text)

X_held = df_held['content']
y_held = df_held['label']

#extract features from held-out set
print('------------Feature Extraction------------')

analyzer = SentimentIntensityAnalyzer()

#3.1 sentiment features
def get_vader_sentiment(text):
    vader_scores = analyzer.polarity_scores(text)
    blob = TextBlob(text)
    return {
        'vader_pos'           : vader_scores['pos'],
        'vader_neu'           : vader_scores['neu'],
        'vader_neg'           : vader_scores['neg'],
        'vader_compound'      : vader_scores['compound'],
        'polarity'            : blob.sentiment.polarity,
        'subjectivity'        : blob.sentiment.subjectivity,
        'emotional_intensity' : abs(vader_scores['compound'])
    }

held_sentiment = X_held.apply(get_vader_sentiment)
held_sentiment_df = pd.DataFrame(list(held_sentiment))

print(f"Sentiment features shape : {held_sentiment_df.shape}")

#3.2 TF-IDF features
held_tfidf = tfidf.transform(X_held)

print(f"TF-IDF features shape    : {held_tfidf.shape}")

#3.2 style features
def extract_style_features(text):
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    words     = text.split()

    avg_sentence_length = len(words) / max(len(sentences), 1)

    punct_count = sum(1 for c in text if c in '.,!?;:')
    punct_ratio = punct_count / max(len(text), 1)

    capital_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    capital_ratio = capital_words / max(len(words), 1)

    try:
        readability = textstat.flesch_reading_ease(text)
    except:
        readability = 0.0

    return {
        'avg_sentence_length' : avg_sentence_length,
        'punct_ratio'         : punct_ratio,
        'capital_word_ratio'  : capital_ratio,
        'readability_score'   : readability
    }

held_style    = X_held.apply(extract_style_features)
held_style_df = pd.DataFrame(list(held_style))

print(f"Style features shape     : {held_style_df.shape}")

#3.3 Credibility features
credible_domains = {
    'reuters.com', 'bbc.com', 'bbc.co.uk', 'apnews.com',
    'theguardian.com', 'nytimes.com', 'washingtonpost.com',
    'npr.org', 'bloomberg.com', 'economist.com'
}

def get_credibility_features(text):
    text_lower = text.lower()
    known_source   = int(any(domain in text_lower for domain in credible_domains))
    author_present = int(
        'by '        in text_lower or
        'author'     in text_lower or
        'reporter'   in text_lower or
        'journalist' in text_lower
    )
    date_pattern     = r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4})\b'
    pub_date_present = int(bool(re.search(date_pattern, text_lower)))
    citation_count   = text_lower.count('according to') + text_lower.count('said') + text_lower.count('reported')

    return {
        'known_source_flag' : known_source,
        'author_present'    : author_present,
        'pub_date_present'  : pub_date_present,
        'citation_count'    : citation_count
    }

held_cred    = X_held.apply(get_credibility_features)
held_cred_df = pd.DataFrame(list(held_cred))

print(f"Credibility features shape : {held_cred_df.shape}")



# Combine all features 
held_sentiment_sparse = sp.csr_matrix(held_sentiment_df.values)
held_style_sparse     = sp.csr_matrix(held_style_df.values)
held_cred_sparse      = sp.csr_matrix(held_cred_df.values)

X_held_final = sp.hstack([
    held_tfidf,
    held_sentiment_sparse,
    held_style_sparse,
    held_cred_sparse
])

print(f"\nFinal held-out feature matrix : {X_held_final.shape}")


#  Combine all features for NB (scaled) 
held_dense = np.hstack([
    held_sentiment_df.values,
    held_style_df.values,
    held_cred_df.values
])

# Use already fitted scaler — DO NOT refit
held_dense_scaled = scaler.transform(held_dense)

X_held_nb = sp.hstack([
    held_tfidf,
    sp.csr_matrix(held_dense_scaled)
])

print(f"NB held-out feature matrix    : {X_held_nb.shape}")

#predictions
print('------------Predictions------------')

# Logistic Regression predictions
lr_pred_held = lr_model.predict(X_held_final)
lr_prob_held = lr_model.predict_proba(X_held_final)[:, 1]

# Naive Bayes predictions
nb_pred_held = nb_model.predict(X_held_nb)
nb_prob_held = nb_model.predict_proba(X_held_nb)[:, 1]

#Output
print('--------------Output--------------')

# Build results dataframe
results_df = pd.DataFrame({
    'title'           : df_held['title'].values,
    'true_label'      : y_held.values,
    'true_verdict'    : y_held.map({0: 'FAKE', 1: 'REAL'}).values,

    # Logistic Regression
    'lr_prediction'   : lr_pred_held,
    'lr_verdict'      : pd.Series(lr_pred_held).map({0: 'FAKE', 1: 'REAL'}).values,
    'lr_confidence'   : (lr_prob_held * 100).round(2),

    # Naive Bayes
    'nb_prediction'   : nb_pred_held,
    'nb_verdict'      : pd.Series(nb_pred_held).map({0: 'FAKE', 1: 'REAL'}).values,
    'nb_confidence'   : (nb_prob_held * 100).round(2),
})


#correct/wrong flags
results_df['lr_correct'] = results_df['lr_prediction'] == results_df['true_label']
results_df['nb_correct'] = results_df['nb_prediction'] == results_df['true_label']

#print sample predictions
print('-------------Sample Predictions------------')

for i, row in results_df.head(10).iterrows():
    print(f"\nArticle {i+1}: {row['title'][:60]}...")
    print(f"  True Label : {row['true_verdict']}")
    print(f"  LR Verdict : {row['lr_verdict']}  (Confidence: {row['lr_confidence']}% Real)")
    print(f"  NB Verdict : {row['nb_verdict']}  (Confidence: {row['nb_confidence']}% Real)")
    print(f"  LR Correct : {'Correct' if row['lr_correct'] else 'Wrong'}")
    print(f"  NB Correct : {'Correct' if row['nb_correct'] else 'Wrong'}")