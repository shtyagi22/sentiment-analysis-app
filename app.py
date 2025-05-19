import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

MODEL_PATH = "model"

print("🚀 Starting Sentiment Analyzer App...")

# === Load Model & Tokenizer with error handling ===
try:
    print(f"🔍 Loading model from: {MODEL_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    print("❌ Failed to load model or tokenizer.")
    print("Error:", e)
    raise SystemExit("🛑 Exiting due to model loading error.")

# === Define the sentiment analysis function ===
def analyze_sentiment(text):
    print(f"\n📩 Received input: {text}")
    if not text or not text.strip():
        print("⚠️ Empty input received.")
        return "Please enter a valid review."

    try:
        result = sentiment_pipeline(text)[0]
        label = result['label']
        score = round(result['score'] * 100, 2)
        emoji = "😊" if label == "POSITIVE" else "😠"
        output = f"{emoji} {label} ({score}%)"
        print(f"✅ Sentiment prediction: {output}")
        return output
    except Exception as e:
        print("❌ Error during prediction.")
        print("Error:", e)
        return "An error occurred during prediction."

# === Launch Gradio Interface ===
print("🚀 Launching Gradio interface...")
gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Enter a movie review..."),
    outputs="text",
    title="🎬 IMDb Sentiment Analyzer",
    description="Enter a movie review to classify it as POSITIVE or NEGATIVE using a fine-tuned DistilBERT model.",
    theme="default"
).launch()
