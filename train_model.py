import pandas as pd
from sklearn.model_selection import train_test_split

# קריאה לקובץ הדאטה עם מפריד ; ושמות עמודות
df = pd.read_csv("data/train.txt", sep=";", names=["text", "label"])
print(df.head())

# פיצול לסט אימון וסט בדיקה
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.label)

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# בניית pipeline: וקטוריזציה + נרמול + סיווג
pipeline = Pipeline([
    ("vect", CountVectorizer()),
    ("tfidf", TfidfTransformer()),
    ("clf", MultinomialNB()),
])

# אימון המודל
pipeline.fit(train_df.text, train_df.label)

# בדיקת דיוק
accuracy = pipeline.score(test_df.text, test_df.label)
print(f"📊 דיוק המודל: {accuracy:.2%}")

# שמירת המודל לקובץ
joblib.dump(pipeline, "emoji_mood_model.joblib")
print("✅ המודל נשמר כ־emoji_mood_model.joblib")
