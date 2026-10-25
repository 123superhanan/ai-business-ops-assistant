#%%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

#%%
df = pd.read_csv("../data/tickets.csv")
print(df.head())
print(df.info())




# %%
print(df['category'].value_counts())

# %%
X = df['ticket_text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%

pipeline = Pipeline([
    ('tfidf',TfidfVectorizer(stop_words='english', lowercase=True)),
    ('classifier', LogisticRegression(random_state=42))
])
cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=3,
    scoring='accuracy'
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(
    f" 3-Fold Cross-Validation Accuracy: "
    f"{cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})"
)
#%%
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("\nSenior-Level Evaluation Report:")
print(classification_report(y_test, y_pred))
# %%
print("Full dataset:")
print(df["category"].value_counts())

print("\nTraining set:")
print(y_train.value_counts())

print("\nTest set:")
print(y_test.value_counts())
# %%
