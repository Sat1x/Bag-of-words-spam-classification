import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Load the dataset
df = pd.read_csv('spam.csv')

# Create a binary target variable: 1 for spam, 0 for ham
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Basic text preprocessing
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    return text

df['processed_message'] = df['Message'].apply(preprocess_text)

# Add text length as a feature
df['text_length'] = df['Message'].str.len()
df['word_count'] = df['Message'].str.split().str.len()

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(df.processed_message, df.spam, test_size=0.2, random_state=42)

# Define multiple models
models = {
    'Naive Bayes (Count)': Pipeline([
        ('vectorizer', CountVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ]),
    'Naive Bayes (TF-IDF)': Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ]),
    'Logistic Regression': Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

# Train and evaluate all models
results = {}
for name, model in models.items():
    print(f"\n=== {name} ===")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'accuracy': accuracy,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Create visualizations
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Accuracy comparison
accuracies = [results[name]['accuracy'] for name in results.keys()]
axes[0, 0].bar(results.keys(), accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Classification report heatmap for best model
best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
report = results[best_model]['report']
report_df = pd.DataFrame(report).transpose()
report_df = report_df.drop('support', axis=1)
sns.heatmap(report_df, annot=True, cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title(f'Classification Report - {best_model}', fontsize=14, fontweight='bold')

# 3. Confusion matrix for best model
cm = confusion_matrix(y_test, results[best_model]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[1, 0])
axes[1, 0].set_title(f'Confusion Matrix - {best_model}', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# 4. Text length distribution by class
axes[1, 1].hist(df[df['spam'] == 0]['text_length'], alpha=0.7, label='Ham', bins=30, color='blue')
axes[1, 1].hist(df[df['spam'] == 1]['text_length'], alpha=0.7, label='Spam', bins=30, color='red')
axes[1, 1].set_title('Text Length Distribution by Class', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Text Length')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('spam_classification_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Create word frequency analysis for spam vs ham
def get_top_words(text_series, n=10):
    all_words = []
    for text in text_series:
        words = text.split()
        all_words.extend(words)
    return Counter(all_words).most_common(n)

# Get top words for spam and ham
spam_words = get_top_words(df[df['spam'] == 1]['processed_message'], 10)
ham_words = get_top_words(df[df['spam'] == 0]['processed_message'], 10)

# Create word frequency comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Spam word frequency
spam_words_df = pd.DataFrame(spam_words, columns=['word', 'count'])
sns.barplot(data=spam_words_df, x='count', y='word', ax=ax1, palette='Reds_r')
ax1.set_title('Most Frequent Words in Spam Messages', fontsize=14, fontweight='bold')
ax1.set_xlabel('Frequency')

# Ham word frequency
ham_words_df = pd.DataFrame(ham_words, columns=['word', 'count'])
sns.barplot(data=ham_words_df, x='count', y='word', ax=ax2, palette='Blues_r')
ax2.set_title('Most Frequent Words in Ham Messages', fontsize=14, fontweight='bold')
ax2.set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('spam_ham_word_frequency.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nBest performing model: {best_model} with accuracy: {results[best_model]['accuracy']:.4f}")
print("Visualizations saved as 'spam_classification_results.png' and 'spam_ham_word_frequency.png'")
