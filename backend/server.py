from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load GPT-2 model and tokenizer
print("Loading GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
if torch.cuda.is_available():
    model.to('cuda')
print("GPT-2 model loaded.")

# Sample training data for decision tree
messages = [
    "hello",
    "buy a book",
    "return a product",
    "thank you",
    "help me",
    "shipping information",
    "I need a refund",
    "what genres do you have?",
    "how to track my order?",
    "can I change my shipping address?",
    "I'd like to check the location of my order.",
    "Hi there! I'd like to check the location of my order.",
    "What is the status of my order?",
    "Can you tell me where my package is?"
]
labels = [1, 2, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 13]

# Vectorize messages using TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(messages)
y_train = labels

# Initialize and train the decision tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Define response mapping based on decision tree labels
response_mapping = {
    1: "Hi there! How can I assist you today?",
    2: "Sure, I can help you buy a book. What genre are you interested in?",
    3: "I'm sorry to hear you want to return a product. Can you provide more details?",
    4: "You're welcome! If you have any more questions, feel free to ask.",
    5: "Of course! What do you need help with?",
    6: "We offer free shipping on orders over $50. Delivery typically takes 3-5 business days.",
    7: "You can request a refund within 30 days of purchase. Please visit our returns page for more details.",
    8: "You can track your order by clicking on the 'Orders' section in your account.",
    9: "Yes, you can change your shipping address by updating your account settings.",
    10: "Your order was shipped on October 10th and is currently in transit. The estimated delivery date is October 18th. Would you like to see the tracking details?",
    11: "Your order is currently in transit. The estimated delivery date is October 18th.",
    12: "The status of your order is currently in transit. Please check your account for tracking details.",
    13: "Your package is on its way! The estimated delivery date is October 18th."
}

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    conversation_history = data.get('conversation', [])

    if not user_message:
        return jsonify({'reply': "Please provide a message.", 'conversation': conversation_history}), 400

    # Vectorize the user message
    X_test = vectorizer.transform([user_message])
    label = decision_tree.predict(X_test)[0]

    # Get response from mapping if it is a specific query
    if label in response_mapping:
        response = response_mapping[label]
    else:
        # Otherwise, generate response using GPT-2
        response = generate_gpt2_response(conversation_history, user_message)

    # Update conversation history with user and bot message
    conversation_history.append({'role': 'user', 'content': user_message})
    conversation_history.append({'role': 'bot', 'content': response})

    return jsonify({'reply': response, 'conversation': conversation_history})

def generate_gpt2_response(conversation_history, user_message):
    # Combine the conversation history into a single string
    conversation_text = ""
    for message in conversation_history:
        conversation_text += f"{message['role']}: {message['content']}\n"

    # Add the new user message to the conversation
    conversation_text += f"User: {user_message}\n"

    # Tokenize the conversation and generate response with GPT-2
    inputs = tokenizer.encode(conversation_text, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response[len(conversation_text):].strip()
    return response if response else "I'm here to help!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
