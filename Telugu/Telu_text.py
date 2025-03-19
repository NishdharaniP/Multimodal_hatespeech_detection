import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import re

# Preprocessing function for Telugu text
def preprocess_telugu_text(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Load training dataset
train_file_path = r'C:\Users\nishd\OneDrive\Documents\shared task\multimodal-hatespeech\multimodal-hatespeech\train\telugu\text\TE-AT-train.xlsx'
df_train = pd.read_excel(train_file_path)

# Apply preprocessing to training data
df_train['Transcript'] = df_train['Transcript'].apply(preprocess_telugu_text)

# Save preprocessed training data
preprocessed_train_path = r'C:\Users\nishd\OneDrive\Documents\shared task\preprocessedfiles\telugu_randompreprocess.tsv'
df_train.to_csv(preprocessed_train_path, sep='\t', index=False)

# Vectorization: TF-IDF for text data
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = tfidf_vectorizer.fit_transform(df_train['Transcript'])

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
y = df_train['Class Label Short']

# Label encode the target variable
y_encoded = LabelEncoder().fit_transform(y)

# Apply SMOTE
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y_encoded)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("Classification Report:\n", classification_report(y_test, y_pred))

# Load test dataset (without labels)
test_file_path = r'C:\Users\nishd\OneDrive\Documents\shared task\multimodal-hatespeech\multimodal-hatespeech\test (1)\test\telugu\text\TE-AT-test.xlsx'
df_test = pd.read_excel(test_file_path)

# Preprocess the test data
df_test['Transcript'] = df_test['Transcript'].apply(preprocess_telugu_text)

# Vectorize the test data using the same TF-IDF vectorizer
X_test_tfidf = tfidf_vectorizer.transform(df_test['Transcript'])

# Make predictions on the test data
y_test_pred = rf_model.predict(X_test_tfidf)

# Decode the predicted labels back to the original classes
# Use the same LabelEncoder instance used during training
y_test_pred_labels = LabelEncoder().fit(y).inverse_transform(y_test_pred)


# Store the predictions in the test DataFrame
df_test['Predicted Label'] = y_test_pred_labels

# Save the predictions to a .tsv file
prediction_file_path = r'C:\Users\nishd\OneDrive\Documents\shared task\results\telugu_randomprediction.tsv'
df_test.to_csv(prediction_file_path, sep='\t', index=False)

print(f"Preprocessed data saved to '{preprocessed_train_path}'")
print(f"Predictions saved to '{prediction_file_path}'")