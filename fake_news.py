# import pandas as pd
# pd.set_option('display.max_columns', None)

# # Fix: use quotechar and quoting to handle embedded commas/quotes in text fields
# df = pd.read_csv(
#     'WELFake_Dataset.csv',
#     engine='python',
#     quotechar='"',
#     on_bad_lines='warn'   # warn instead of skip — shows you what's still breaking
# )

# print(df.shape)                        # should show (72151, 4)
# print(df['label'].value_counts())
# print(df.head())












import csv
from pydoc import text
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
import scipy.sparse as sp









# Show all columns
pd.set_option('display.max_columns', None)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#  1  DATA INGESTION

print("=" * 50)
print(" 1 — Data Ingestion")
print("=" * 50)

# 1.1: Fix encoding issues (curly quotes) 
print("\nStep 1.1: Fixing encoding issues...")

with open('WELFake_Dataset.csv', 'rb') as f:
    raw = f.read()

raw = raw.replace(b'\xe2\x80\x9c', b"'").replace(b'\xe2\x80\x9d', b"'")
raw = raw.replace(b'\xe2\x80\x99', b"'")

with open('welfake_fixed.csv', 'wb') as f:
    f.write(raw)

print("Encoding fixed and saved to welfake_fixed.csv")

# 1.2: Parse CSV safely 
print("\nStep 1.2: Parsing CSV safely...")

records = []

with open('welfake_fixed.csv', 'r', encoding='utf-8-sig', errors='replace', newline='') as f:
    reader = csv.reader(f, quotechar='"', doublequote=True, skipinitialspace=True)
    next(reader)  # skip header

    for row in reader:
        # Remove trailing empty columns caused by extra commas
        while row and row[-1].strip() == '':
            row.pop()

        if len(row) < 4:
            continue

        label = row[-1].strip()

        # Keep only valid labels
        if label not in ('0', '1'):
            continue

        records.append({
            'idx':   row[0].strip(),
            'title': row[1].strip(),
            'text':  ','.join(row[2:-1]).strip(),
            'label': int(label)
        })

df = pd.DataFrame(records)

print(f"Raw shape: {df.shape}")
print(df['label'].value_counts())

#  1.3: Handle nulls 
print("\nStep 1.3: Handling nulls...")

df['title'] = df['title'].replace('', pd.NA)
df['title'] = df['title'].fillna('')
df = df.dropna(subset=['text', 'label'])

print(f"After null handling: {df.shape}")

#  1.4: Merge title + text 
print("\nStep 1.4: Merging title and text...")

df['content'] = (df['title'].str.strip() + ' ' + df['text'].str.strip()).str.strip()

#  1.5: Remove duplicates 
print("\nStep 1.5: Removing duplicates...")

before = len(df)
df = df.drop_duplicates(subset=['content'])

print(f"Duplicates removed : {before - len(df)}")
print(f"Remaining rows     : {len(df)}")

#  1.6: Stratified split — 60k train, rest held-out 
print("\nStep 1.6: Stratified split into 60k training + held-out...")

df_60k, df_held_out = train_test_split(
    df,
    train_size=60000,
    random_state=42,
    stratify=df['label']
)

df_60k      = df_60k.reset_index(drop=True)
df_held_out = df_held_out.reset_index(drop=True)

print(f"Training pool : {len(df_60k)}")
print(f"Held-out test : {len(df_held_out)}")
print(f"\nTraining label distribution:\n{df_60k['label'].value_counts()}")
print(f"\nHeld-out label distribution:\n{df_held_out['label'].value_counts()}")

#  1.7: Save outputs 
print("\nStep 1.7: Saving outputs...")

df_60k.to_csv('stage1_output.csv', index=False)
df_held_out.to_csv('held_out_test.csv', index=False)

print("\nStage 1 complete.")
print(f"Training data : stage1_output.csv → {df_60k.shape}")
print(f"Held-out test : held_out_test.csv → {df_held_out.shape}")










#   2 .PREPROCESSING

print("\n" + "=" * 50)
print("STAGE 2 — Preprocessing")
print("=" * 50)

#  2.1: Lowercasing 
print("\nStep 2.1: Lowercasing...")

df_60k['content'] = df_60k['content'].str.lower()
print("Lowercasing done.")

#  2.2: Remove punctuation & special characters 
print("\nStep 2.2: Removing punctuation & special characters...")

def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)          # keep only letters & spaces
    text = re.sub(r'\s+', ' ', text).strip()      # collapse extra whitespace
    return text

df_60k['content'] = df_60k['content'].apply(clean_text)
print("Noise removal done.")

#  2.3: Remove stopwords 
print("\nStep 2.3: Removing stopwords...")

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df_60k['content'] = df_60k['content'].apply(remove_stopwords)
print("Stopwords removed.")

# 2.4: Lemmatisation 
print("\n Lemmatising...")

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = text.split()
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(lemmatized)

df_60k['content'] = df_60k['content'].apply(lemmatize_text)
print("Lemmatisation done.")

# 2.5: Train / Test split
print("\nStep 2.5: Train/Test split...")

X = df_60k['content']
y = df_60k['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nStage 2 complete.")
print(f"Training samples : {X_train.shape[0]}")   
print(f"Test samples     : {X_test.shape[0]}")    
print(f"\nTrain label distribution:\n{y_train.value_counts()}")
print(f"\nTest label distribution:\n{y_test.value_counts()}")

# ── Optional: Save preprocessed splits ───────────────────────
# train_df = pd.DataFrame({'content': X_train, 'label': y_train})
# test_df  = pd.DataFrame({'content': X_test,  'label': y_test})
# train_df.to_csv('stage2_train.csv', index=False)
# test_df.to_csv('stage2_test.csv',   index=False)




#3. feature extraction

print('------------Feature Extraction------------')

#3.1: Sentiment Analysis (VADER) Valence Aware Dictionary and sEntiment Reasoner

analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    vader_scores = analyzer.polarity_scores(text)
    blob = TextBlob(text)
    
    return {
        # VADER scores
        'vader_pos': vader_scores['pos'],
        'vader_neu': vader_scores['neu'],
        'vader_neg': vader_scores['neg'],
        'vader_compound': vader_scores['compound'],

        # TextBlob scores
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,

        #emotional tone (simple heuristic)
        'emotional_intensity': abs(vader_scores['compound'])
    }

print("\nExtracting sentiment features...")
train_sentiment = X_train.apply(get_vader_sentiment)
train_sentiment_df = pd.DataFrame(list(train_sentiment))

print('Extracting sentiment features for test set...')
test_sentiment = X_test.apply(get_vader_sentiment)
test_sentiment_df = pd.DataFrame(list(test_sentiment))

print(f"Sentiment features shape (train) : {train_sentiment_df.shape}")
print(f"Sentiment features shape (test)  : {test_sentiment_df.shape}")
print(f"\nSample sentiment features:\n{train_sentiment_df.head()}")


#3.2: writing style features

print('Writing style features...')

print('Fitting TF-IDF vectorizer')

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.95)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"TF-IDF features shape (train) : {X_train_tfidf.shape}")
print(f"TF-IDF features shape (test)  : {X_test_tfidf.shape}")

#readability and style features
print('Extracting Style Features')

def extract_style_features(text):
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()

    #Average sentence length
    avg_sentence_length = len(words)/ max(len(sentences), 1)

    #Punctuation ratio
    punct_count = sum(1 for c in text if c in '.,!?;:')
    punct_ratio = punct_count / max(len(text), 1)

    #capital word ratio
    capital_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    capital_ratio = capital_words / max(len(words), 1)

    #readability scores
    try:
        readability = textstat.flesch_reading_ease(text)
    except:
        readability = 0.0

    return {
        'avg_sentence_length': avg_sentence_length,
        'punct_ratio': punct_ratio,
        'capital_word_ratio': capital_ratio,
        'readability_score': readability
    }

train_style = X_train.apply(extract_style_features)
train_style_df = pd.DataFrame(list(train_style))

test_style = X_test.apply(extract_style_features)
test_style_df = pd.DataFrame(list(test_style))

print(f"Style features shape (train) : {train_style_df.shape}")
print(f"Style features shape (test)  : {test_style_df.shape}")
print(f"\nSample style features:\n{train_style_df.head()}")

#3.3 credibility features (simple heuristics)
print('Extracting credibility features...')

credible_domains = {
    'reuters.com', 'bbc.com', 'bbc.co.uk', 'apnews.com',
    'theguardian.com', 'nytimes.com', 'washingtonpost.com',
    'npr.org', 'bloomberg.com', 'economist.com'
}

def get_credibility_features(text):
    text = text.lower()
    known_source = int(any(domain in text for domain in credible_domains))
    author_present = int(
        'by' in text or
        'author' in text or
        'reporter' in text or
        'journalist' in text
    )

    import re
    date_pattern = r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4})\b'
    pub_date_present = int(bool(re.search(date_pattern, text )))

    #citation count
    citation_count = text.count('according to') + text.count('said')+text.count('reported')

    return {
        'known_source_flag'  : known_source,
        'author_present'     : author_present,
        'pub_date_present'   : pub_date_present,
        'citation_count'     : citation_count
    }

train_cred = X_train.apply(get_credibility_features)
train_cred_df = pd.DataFrame(list(train_cred))

test_cred = X_test.apply(get_credibility_features)
test_cred_df = pd.DataFrame(list(test_cred))

print(f"Sample Credibility Features: {train_cred_df.head()}")

#combine all features
print('Combine Features')

train_sentiment_sparse = sp.csr_matrix(train_sentiment_df.values)
test_sentiment_sparse = sp.csr_matrix(test_sentiment_df.values)

train_style_sparse = sp.csr_matrix(train_style_df.values)
test_style_sparse = sp.csr_matrix(test_style_df.values)

train_cred_sparse      = sp.csr_matrix(train_cred_df.values)
test_cred_sparse       = sp.csr_matrix(test_cred_df.values)

X_train_final = sp.hstack([
    X_train_tfidf,          # 3b: TF-IDF (10,000 features)
    train_sentiment_sparse, # 3a: Sentiment (7 features)
    train_style_sparse,     # 3b: Style (4 features)
    train_cred_sparse       # 3c: Credibility (4 features)
])

X_test_final = sp.hstack([
    X_test_tfidf,
    test_sentiment_sparse,
    test_style_sparse,
    test_cred_sparse
])

print(f"\nFinal feature matrix shape (train) : {X_train_final.shape}")
print(f"Final feature matrix shape (test)  : {X_test_final.shape}")
print(f"\nFeature breakdown:")
print(f"  TF-IDF features      : 10,000")
print(f"  Sentiment features   : {train_sentiment_df.shape[1]}")
print(f"  Style features       : {train_style_df.shape[1]}")
print(f"  Credibility features : {train_cred_df.shape[1]}")
print(f"  Total features       : {X_train_final.shape[1]}")






#model building
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
import joblib
import time

print('------------Model Building------------')

scaler = MinMaxScaler()

#scale dense features
train_dense = np.hstack([train_sentiment_df.values, train_style_df.values, train_cred_df.values])

test_dense = np.hstack([test_sentiment_df.values, test_style_df.values, test_cred_df.values])

train_dense_scaled = scaler.fit_transform(train_dense)
test_dense_scaled  = scaler.transform(test_dense)

X_train_nb = sp.hstack([X_train_tfidf, sp.csr_matrix(train_dense_scaled)])
X_test_nb =  sp.hstack([X_test_tfidf, sp.csr_matrix(test_dense_scaled)])

print(f"Feature matrix for Naive Bayes (train) : {X_train_nb.shape}")
print(f"Feature matrix for Naive Bayes (test)  : {X_test_nb.shape}")

#Evaluate model
def evaluate_model(model_name, y_true, y_pred, y_prob):
    print(f"-------{model_name} Results-----------")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"  F1 Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"  AUC-ROC   : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Fake', 'Real']))
    return {
        'accuracy' : accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall'   : recall_score(y_true, y_pred),
        'f1'       : f1_score(y_true, y_pred),
        'auc_roc'  : roc_auc_score(y_true, y_prob)
    }


#4.1 logistic regression

print('Logistic Regression')

start = time.time()

model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42, n_jobs=-1)

model.fit(X_train_final, y_train)
model_train_time = time.time() - start

print(f"Training time : {model_train_time:.2f} seconds")

#predictions
lr_pred = model.predict(X_test_final)
lr_prob = model.predict_proba(X_test_final)[:, 1]

#Evaluate
lr_results = evaluate_model("Logistic Regression",y_test, lr_pred, lr_prob)

# Save model
joblib.dump(model, 'lr_model.pkl')
print("\nLogistic Regression model saved → lr_model.pkl")


#4.2 Naive bayes
print('Naive Bayes')

start = time.time()

nb_model = MultinomialNB(alpha=1.0)

nb_model.fit(X_train_nb, y_train)
nb_train_time = time.time() - start

print(f"Training time : {nb_train_time:.2f} seconds")

# Predictions
nb_pred = nb_model.predict(X_test_nb)
nb_prob = nb_model.predict_proba(X_test_nb)[:, 1]

# Evaluate
nb_results = evaluate_model("Naive Bayes",y_test, nb_pred, nb_prob)

# Save model
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(scaler,   'nb_scaler.pkl')
print("\nNaive Bayes model saved → nb_model.pkl")
print("Scaler saved           → nb_scaler.pkl")

#model comparison

print('Model Comparison')

comparison_df = pd.DataFrame({
    'Metric'              : ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'],
    'Logistic Regression' : [
        f"{lr_results['accuracy']:.4f}",
        f"{lr_results['precision']:.4f}",
        f"{lr_results['recall']:.4f}",
        f"{lr_results['f1']:.4f}",
        f"{lr_results['auc_roc']:.4f}"
    ],
    'Naive Bayes'         : [
        f"{nb_results['accuracy']:.4f}",
        f"{nb_results['precision']:.4f}",
        f"{nb_results['recall']:.4f}",
        f"{nb_results['f1']:.4f}",
        f"{nb_results['auc_roc']:.4f}"
    ]
})

print(comparison_df.to_string(index=False))

print(f"\nTraining Time:")
print(f"  Logistic Regression : {model_train_time:.2f} sec")
print(f"  Naive Bayes         : {nb_train_time:.2f} sec")

# Best model selection
best_model_name = "Logistic Regression" if lr_results['f1'] >= nb_results['f1'] else "Naive Bayes"
best_model      = model if lr_results['f1'] >= nb_results['f1'] else nb_model

print(f"\nBest model (by F1 Score) : {best_model_name}")

# Save best model
joblib.dump(best_model, 'best_model.pkl')
print(f"Best model saved → best_model.pkl")





joblib.dump(tfidf,    'tfidf_vectorizer.pkl')  # TF-IDF vectorizer
joblib.dump(scaler,   'nb_scaler.pkl')          # Scaler for NB
joblib.dump(model,    'lr_model.pkl')           # Logistic Regression
joblib.dump(nb_model, 'nb_model.pkl')           # Naive Bayes
joblib.dump(best_model, 'best_model.pkl')       # Best model
































#Preprocessing steps:
#1. Load the dataset and handle encoding issues.
#2. Parse the CSV safely, accounting for embedded commas and quotes.
#3. Handle null values in 'title' and 'text' columns.
#4. Merge 'title' and 'text' into a single 'content' column.
#5. Remove duplicate entries based on 'content'.
#6. Perform stratified sampling to create a balanced training set of 70,000 records.
'''
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#2.1 lowercasing
df_70k['content'] = df_70k['content'].str.lower()

#2.2 remove punctuation and special characters
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)              # keep only letters & spaces
    text = re.sub(r'\s+', ' ', text).strip()          # collapse extra whitespace
    return text

df_70k['content'] = df_70k['content'].apply(clean_text)

#remove stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()
    tokens= [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df_70k['content'] = df_70k['content'].apply(remove_stopwords) 

#2.4 lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = text.split()
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(lemmatized)

df_70k['content'] = df_70k['content'].apply(lemmatize_text)

#train-test split
X = df_70k['content']
y = df_70k['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")'''
