import nltk
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import torch

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

# Define a list of user inputs and corresponding bot responses
user_inputs = [
    "I feel sad",
    "I'm going through a tough time",
    "I don't know what to do with my life",
    "I'm feeling anxious",
    "I'm struggling with my relationships",
    "I'm not happy with my job",
]

bot_responses = [
    "I'm sorry to hear that. Can you tell me more about what's been bothering you?",
    "It sounds like you're going through a difficult time. How long have you been feeling this way?",
    "Sometimes it helps to talk about our feelings. What specifically has been bothering you?",
    "Anxiety can be tough to deal with. Have you tried any relaxation techniques?",
    "Relationships can be challenging. What specifically is causing you trouble?",
    "Job dissatisfaction is common. Have you considered talking to your supervisor or exploring other opportunities?",
]

# Define a list of riddles and jokes
riddles = [
    {
        "question": "I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?",
        "answer": "echo"
    },
    {
        "question": "What has keys but can't open locks?",
        "answer": "piano"
    },
    {
        "question": "What has a heart that doesn't beat?",
        "answer": "artichoke"
    }
]

jokes = [
    "Why don't scientists trust atoms? Because they make up everything!",
    "Why don't skeletons fight each other? They don't have the guts!",
    "Why did the scarecrow win an award? Because he was outstanding in his field!",
]


# Create a function to preprocess and vectorize the user inputs
def preprocess_inputs(inputs):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(inputs)
    return vectorizer


# Create a function to generate bot responses using cosine similarity
def generate_response(user_input, vectorizer):
    user_input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(
        user_input_vector, vectorizer.transform(user_inputs))
    most_similar_index = np.argmax(similarities)
    return bot_responses[most_similar_index]


# Create a function to simulate the chatbot
def chatbot():
    print("Welcome to the chatbot. How can I assist you today?")
    vectorizer = preprocess_inputs(user_inputs)
    while True:
        user_input = input("> ")
        if user_input.lower() == "who are you":
            print("I am a chatbot designed to assist you. How can I help?")
        elif user_input.lower() == "how are you":
            print(
                "Thank you for asking. I'm an AI, so I don't have feelings, but I'm here to assist you.")
        elif user_input.lower() == "tell me something interesting":
            print("Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!")
        elif user_input.lower() == "tell me a riddle":
            riddle = random.choice(riddles)
            print("Here's a riddle for you:")
            print(riddle["question"])
            user_answer = input("Your answer: ")
            if user_answer.lower() == riddle["answer"]:
                print("Congratulations! You got it right!")
            else:
                print("Oops! That's incorrect. The correct answer is:",
                      riddle["answer"])
        elif user_input.lower() == "tell me a joke":
            joke = random.choice(jokes)
            print("Here's a joke for you:")
            print(joke)
        else:
            bot_response = generate_response(user_input, vectorizer)
            print(bot_response)

        if user_input.lower() == "quit":
            print("Thank you for chatting. Take care!")
            break


# Run the chatbot
if __name__ == "__main__":
    chatbot()
