from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai
import os
from dotenv import load_dotenv

def load_finbert_model():
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        return tokenizer, model

def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    sentiment = torch.softmax(logits, dim=-1).numpy()
    return sentiment



def generate_investment_advice(prompt, api_key):
    openai.api_key = api_key
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()


def investment_advice_engine(news_article, api_key):
    # Load RoBERTa model and tokenizer
    tokenizer, model = load_finbert_model()

    # Analyze sentiment
    sentiment = analyze_sentiment(news_article, tokenizer, model)

    # Generate a prompt for GPT-4 based on sentiment analysis
    sentiment_str = f"positive sentiment: {sentiment[0][1]:.2f}, negative sentiment: {sentiment[0][0]:.2f}"
    prompt = f"Given the following sentiment analysis results: {sentiment_str}. Provide investment advice based on this information."

    # Generate investment advice
    advice = generate_investment_advice(prompt, api_key)

    return advice


# Load the .env.local file from the extracted location
load_dotenv("/Users/joeylam/repo/py/myAIbot/env/.env.local")

# Now you can access the environment variables using os.environ.get()
api_key = os.environ.get('OPENAI_API_KEY')


news_article = "The company's quarterly report shows a significant increase in revenue and market share. Investors are optimistic about the future growth prospects."

advice = investment_advice_engine(news_article, api_key)
print(advice)