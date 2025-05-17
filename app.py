from flask import Flask, request, jsonify
from transformers import pipeline
from googleapiclient.discovery import build

app = Flask(__name__)

# Initialize Emotion Analysis Model
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# YouTube API Configuration
API_KEY = "AIzaSyBXMcNnIxeND-8mSNBB4_dSP4D8kbFtNck"
youtube = build("youtube", "v3", developerKey=API_KEY)

# Emotion-Based Images (Use static paths for web compatibility)
theme_images = {
    "POSITIVE": "/static/positive.jpg",
    "NEUTRAL": "/static/motivation.jpg",
    "NEGATIVE": "/static/hap.jpg"
}

# Updated YouTube Playlists
video_data = {
    "NEGATIVE": "https://www.youtube.com/embed/videoseries?list=PLvx1Z92uzO9PhwqcXbMml-14j-ALCkXVY",
    "NEUTRAL": "https://www.youtube.com/embed/MzpjuBsWIxM",
    "POSITIVE": "https://www.youtube.com/embed/4LFs0YR3xMY"
}

def get_mood(text):
    """Detect user emotion using Hugging Face's model."""
    result = emotion_analyzer(text)
    emotion = result[0]["label"]
    return "POSITIVE" if emotion in ["joy", "love"] else "NEGATIVE" if emotion in ["sadness", "anger", "fear"] else "NEUTRAL"

@app.route("/analyze", methods=["POST"])
def analyze():
    """API endpoint for sentiment analysis and video recommendation."""
    data = request.json
    user_text = data.get("text", "")

    mood = get_mood(user_text)
    video_link = video_data[mood]
    theme_image = theme_images[mood]

    response = {
        "mood": mood,
        "video_link": video_link,
        "theme_image": theme_image
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)