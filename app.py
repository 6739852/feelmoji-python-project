import streamlit as st
import joblib

# טעינת המודל
model = joblib.load("emoji_mood_model.joblib")

# מיפוי רגשות לאימוג'ים
emoji_map = {
    "sadness": "😢",
    "joy": "😄",
    "anger": "😠",
    "love": "😍",
    "surprise": "😲",
    "fear": "😱",
    "neutral": "😐",
}

# הגדרות עמוד
st.set_page_config(page_title="Emoji Mood App", page_icon="🎭", layout="centered")

# כותרת
st.title("🎭 ניחוש רגש עם אימוג’י")
st.write("הכנס משפט או פסקה, וננסה לזהות את הרגש שלך 👇")

# קלט מהמשתמש
text = st.text_area("איך אתה מרגיש היום?", height=150)

# כפתור שליחה
if st.button("🔍 גלה את הרגש"):
    if text.strip():
        prediction = model.predict([text])[0]
        emoji = emoji_map.get(prediction, "❓")
        st.markdown(f"## {emoji}  ({prediction})")
    else:
        st.warning("📌 אנא כתוב טקסט לפני הלחיצה")
