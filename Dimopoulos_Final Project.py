import json
import re
from word2number import w2n
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import speech_recognition as sr
import pyttsx3


nltk.download("punkt")
nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')

data_file = open("intents.json").read()
data = json.loads(data_file)

food_items = ["burger", "pizza", "salad", "pasta", "sandwich"]
food_value = [8.99, 10.99, 6.99, 9.99, 7.99]
words = []
classes = []
data_X = []
data_Y = []


for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        data_X.append(pattern)
        data_Y.append(intent["tag"])

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))


training = []

out_empty = [0] * len(classes)
for idx, doc in enumerate(data_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    output_row = list(out_empty)
    output_row[classes.index(data_Y[idx])] = 1
    training.append([bow, output_row])
random.shuffle(training)
training = np.array(training, dtype=object)
train_X = np.array(list(training[:, 0]))
train_Y = np.array(list(training[:, 1]))


model = Sequential()
model.add(Dense(128, input_shape =(len(train_X[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]), activation="softmax"))


adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_Y, epochs=150, verbose=1)


def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word)
 for word in tokens]
    return tokens


def bag_of_words(text,vocab):
    tokens = clean_text(text)
    bow = [0]*len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)


def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.5
    y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list


def get_response(intents_list, intents_json, food, food_item, qnt):
    if len(intents_list) == 0:
        result = "Sorry! I don't understand."
    else:
        tag = intents_list[0]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == "complete_order":
                result = random.choice(i["responses"])
                order_complete = True
                break
            if food:
                result = random.choice(data["intents"][3]["responses"])  # Select a random response for taking an order
                result = result.replace("[food_item]", food_item)  # Replace the placeholder with the recognized food item
                if qnt != 0:
                    result = result.replace("[quantity]", str(qnt))  # Replace the placeholder with the recognized food item
                order_complete = False
                break
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                order_complete = False
                break
    return result, order_complete


def extract_quantities(sentence):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    # Perform part-of-speech tagging
    tagged_tokens = nltk.pos_tag(tokens)

    # Initialize list to store quantities
    quantities = []

    # Iterate through tagged tokens to identify quantities
    for i in range(len(tagged_tokens)):
        word, pos = tagged_tokens[i]

        # Check if the current word is a cardinal number (CD) or ordinal number (JJ)
        if pos in ['CD', 'JJ']:
            # Append the word to the quantities list
            quantities.append(w2n.word_to_num(word))
    return quantities


def validate_integer_in_range(input_str, min_value, max_value):
    try:
        # Convert input string to an integer
        num = int(input_str)
        # Check if the integer falls within the specified range
        return min_value <= num <= max_value
    except ValueError:
        # If the conversion to integer fails, return False
        return False


def recognize_speech():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        # Recognize speech using Google Speech Recognition
        speech_text = recognizer.recognize_google(audio)
        return speech_text
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))


# Function to speak text
def speak_text(text):
    engine.say(text)
    engine.runAndWait()


food_list = []
quantity_list = []
total_cost = 0.0
order_complete = False
input_device = input("1. Keyboard (Press 0 if you don't want to chat with your chatbot)\n2. Microphone(Speak EXIT if you don't want to chat with your chatbot)\nPlease Select input device:")
while not validate_integer_in_range(input_device, 1, 2):
    input_device = input("Please input choice 1 or 2\n1. Keyboard\n2. Microphone\nPlease Select input device:")


while True:
    food_item = ""
    food = False
    if input_device == "1":
        message = input(" ")
        message.lower()
        if message == "0":
            break
        intents = pred_class(message, words, classes)
        for item in food_items:
            # Check if the food item is mentioned in the user input
            if re.search(r'\b' + re.escape(item) + r'\b', message, re.IGNORECASE):
                intents[0] = "take_order"
                food = True
                food_item = item
                food_list.append(food_item)
                quantity_list += extract_quantities(message)
                if not extract_quantities(message):
                    qnt = 1
                    quantity_list.append(qnt)
                else:
                    qnt = quantity_list[-1]
                total_cost += next((value for value, item in zip(food_value, food_items) if item == food_item), None) * qnt
        if not food:
            qnt = 0
        result, order_complete = get_response(intents, data, food, food_item, qnt)
        print(result)
        if food_list and not order_complete:
            print("Your order now includes: ")
            for i in range(len(food_list)):
                print(food_list[i] + ": " + str(quantity_list[i]))
            print("Your total cost: " + str(round(total_cost, 2)) + "€")
        if order_complete:
            print("To finish your order please provide us with the following:")
            name = input("Name: ")
            last_name = input("Last Name: ")
            address = input("Address: ")
            phone = input("Phone Number: ")
            print("----------Order Details-------------")
            print("Name: " + name)
            print("Last Name: " + last_name)
            print("Address: " + address)
            print("Phone: " + phone)
            for i in range(len(food_list)):
                print(food_list[i] + ": " + str(quantity_list[i]))
            print("Your total cost: " + str(round(total_cost, 2)) + "€")
            print("------------------------------------")
            break
    else:
        # Initialize speech recognition
        recognizer = sr.Recognizer()

        # Initialize text-to-speech engine
        engine = pyttsx3.init()
        speech_input = recognize_speech()
        # Process speech input
        if speech_input:
            print("You said:", speech_input)
            # Perform actions based on speech input
            if speech_input == "exit":
                response = "Terminating ChatBot "
                speak_text(response)
                break
            intents = pred_class(speech_input, words, classes)
            for item in food_items:
                # Check if the food item is mentioned in the user input
                if re.search(r'\b' + re.escape(item) + r'\b', speech_input, re.IGNORECASE):
                    intents[0] = "take_order"
                    food = True
                    food_item = item
                    food_list.append(food_item)
                    quantity_list += extract_quantities(speech_input)
                    if not extract_quantities(speech_input):
                        qnt = 1
                        quantity_list.append(qnt)
                    else:
                        qnt = quantity_list[-1]
                    total_cost += next((value for value, item in zip(food_value, food_items) if item == food_item),
                                       None) * qnt
            if not food:
                qnt = 0
            result, order_complete = get_response(intents, data, food, food_item, qnt)
            print(result)
            speak_text(result)
            if food_list and not order_complete:
                print("Your order now includes: ")
                speak_text("Your order now includes")
                for i in range(len(food_list)):
                    print(food_list[i] + ": " + str(quantity_list[i]))
                    speak_text(food_list[i] + ": " + str(quantity_list[i]))
                print("Your total cost: " + str(round(total_cost, 2)) + "€")
                speak_text("Your total cost: " + str(round(total_cost, 2)) + "€")
            if order_complete:
                print("To finish your order please provide us with the following:")
                speak_text("To finish your order please provide us with the following:")
                information = {}
                prompts = {
                    "name": "name: ",
                    "surname": "Last Name: ",
                    "address": "Address: ",
                    "phone": "Phone Number: "
                }

                for key, prompt in prompts.items():
                    print(prompt)
                    speech_input = recognize_speech()
                    while speech_input is None:
                        print("Please try again.")
                        print(prompt)
                        speech_input = recognize_speech()
                    information[key] = speech_input
                print("----------Order Details-------------")
                print("Name: " + information["name"])
                speak_text("Name" + information["name"])
                print("Last Name: " + information["surname"])
                speak_text("Last Name: " + information["surname"])
                print("Address: " + information["address"])
                speak_text("Address: " + information["address"])
                print("Phone: " + information["phone"])
                speak_text("Phone: " + information["phone"])
                for i in range(len(food_list)):
                    print(food_list[i] + ": " + str(quantity_list[i]))
                print("Your total cost: " + str(round(total_cost, 2)) + "€")
                print("------------------------------------")
                break
