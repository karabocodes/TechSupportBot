# import random
# import json
# import pickle
# import numpy as np
# import tensorflow as tf
# import nltk
# from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()

# intents = json.loads(open('intents.json').read())

# words = []
# classes = []
# documents = []
# ignoreLetters = ['/','?', '.' ,',']

# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         wordList = nltk.word_tokenize(pattern)
#         words.extend(wordList)
#         documents.append((wordList, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
# words = sorted(set(words))

# classes = sorted(set(classes))

# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))

# training = []
# outputEmpty = [0] * len(classes)

# for document in documents:
#     bags = []
#     wordPatterns = document[0]
#     wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
#     for word in words:
#         bags.append(1) if word in wordPatterns else bags.append(0)

#     outputRow = list(outputEmpty)
#     outputRow[classes.index(document[1])] = 1
#     training.append(bags + outputRow)

# random.shuffle(training)
# training = np.array(training)

# trainX = training[:, :len(words)]
# trainY = training[:, len(words):]

# model = tf.keras.Sequential()

# model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# model.save('chatbot_techsupport.h5')
# print("Done")
# import random
# import json
# import pickle
# import numpy as np
# import tensorflow as tf
# import nltk
# from nltk.stem import WordNetLemmatizer
# from sklearn.model_selection import train_test_split

# lemmatizer = WordNetLemmatizer()

# # Load intents from a file
# intents = json.loads(open('intents.json').read())

# # Initialize lists and preprocessing
# words = []
# classes = []
# documents = []
# ignoreLetters = ['/','?', '.' ,',']

# # Tokenize words and prepare training data
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         wordList = nltk.word_tokenize(pattern)
#         words.extend(wordList)
#         documents.append((wordList, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# # Lemmatize and remove duplicates
# words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
# words = sorted(set(words))
# classes = sorted(set(classes))

# # Save preprocessed data
# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))

# # Prepare training data
# training = []
# outputEmpty = [0] * len(classes)

# for document in documents:
#     bags = []
#     wordPatterns = document[0]
#     wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
#     for word in words:
#         bags.append(1) if word in wordPatterns else bags.append(0)

#     outputRow = list(outputEmpty)
#     outputRow[classes.index(document[1])] = 1
#     training.append(bags + outputRow)

# random.shuffle(training)
# training = np.array(training)

# # Split the data into training and testing sets
# trainX, testX, trainY, testY = train_test_split(training[:, :len(words)], training[:, len(words):], test_size=0.2, random_state=42)

# # Build the model
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# # Train the model
# hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=100, validation_data=(testX, testY), verbose=1)

# # Save the trained model
# model.save('chatbot_techsupport.h5')
# print("Done")
# import random
# import json
# import pickle
# import numpy as np
# import tensorflow as tf
# import nltk
# from nltk.stem import WordNetLemmatizer
# from sklearn.model_selection import train_test_split
# from keras.models import load_model

# # Check if GPU is available
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     print("GPU available. Using GPU for training.")
# else:
#     print("No GPU available. Training on CPU.")

# lemmatizer = WordNetLemmatizer()

# # Load intents from a file
# intents = json.loads(open('intents.json').read())

# # Initialize lists and preprocessing
# words = []
# classes = []
# documents = []
# ignoreLetters = ['/','?', '.' ,',']

# # Tokenize words and prepare training data
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         wordList = nltk.word_tokenize(pattern)
#         words.extend(wordList)
#         documents.append((wordList, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# # Lemmatize and remove duplicates
# words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
# words = sorted(set(words))
# classes = sorted(set(classes))

# # Save preprocessed data
# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))

# # Prepare training data
# training = []
# outputEmpty = [0] * len(classes)

# for document in documents:
#     bags = []
#     wordPatterns = document[0]
#     wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
#     for word in words:
#         bags.append(1) if word in wordPatterns else bags.append(0)

#     outputRow = list(outputEmpty)
#     outputRow[classes.index(document[1])] = 1
#     training.append(bags + outputRow)

# random.shuffle(training)
# training = np.array(training)

# # Split the data into training and testing sets
# trainX, testX, trainY, testY = train_test_split(training[:, :len(words)], training[:, len(words):], test_size=0.2, random_state=42)

# # Build the model
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(64, activation='relu'))
# model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# # Experiment with different optimizers
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# # Implement early stopping
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train the model
# epochs = 200
# batch_size = 32  # Experiment with different batch sizes
# hist = model.fit(np.array(trainX), np.array(trainY), epochs=epochs, batch_size=batch_size, 
#                  validation_data=(testX, testY), callbacks=[early_stopping], verbose=1)

# # Save the trained model
# model.save('chatbot_techsupport.h5')
# print("Training completed.")

import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from keras.models import load_model

rl_epochs = 10  # Define the number of RL training epochs (you can adjust this value)

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU available. Using GPU for training.")
else:
    print("No GPU available. Training on CPU.")

# Function to preprocess data
def preprocess_data():
    lemmatizer = WordNetLemmatizer()

    # Load intents from a file
    intents = json.loads(open('intents.json').read())

    # Initialize lists and preprocessing
    words = []
    classes = []
    documents = []
    ignoreLetters = ['/','?', '.' ,',']

    # Tokenize words and prepare training data
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            wordList = nltk.word_tokenize(pattern)
            words.extend(wordList)
            documents.append((wordList, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize and remove duplicates
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
    words = sorted(set(words))
    classes = sorted(set(classes))

    # Save preprocessed data
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    # Prepare training data
    training = []
    outputEmpty = [0] * len(classes)

    for document in documents:
        bags = []
        wordPatterns = document[0]
        wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
        for word in words:
            bags.append(1) if word in wordPatterns else bags.append(0)

        outputRow = list(outputEmpty)
        outputRow[classes.index(document[1])] = 1
        training.append(bags + outputRow)

    random.shuffle(training)
    training = np.array(training)

    # Split the data into training and testing sets
    trainX, testX, trainY, testY = train_test_split(training[:, :len(words)], training[:, len(words):], test_size=0.2, random_state=42)

    return trainX, testX, trainY, testY, words, classes

# Function to build and train the model
def get_user_query():
    # Simple implementation to get a user query from input
    return input("User: ")

def process_user_query(user_query, words, lemmatizer):
    # Simple preprocessing by tokenizing and lemmatizing
    user_query_words = nltk.word_tokenize(user_query)
    user_query_words = [lemmatizer.lemmatize(word.lower()) for word in user_query_words]

    # Create bag-of-words representation
    bag = [1 if word in user_query_words else 0 for word in words]

    return np.array([bag])  # Wrap the bag in an array to match the expected input shape



def get_user_feedback():
    # Simple implementation to get user feedback (reward/penalty)
    feedback = input("Provide feedback (1 for positive, -1 for negative, 0 for neutral): ")
    return int(feedback)

# def update_model(model_response, reward):
#     pass  # Replace with your RL update logic


def build_and_train_model(trainX, testX, trainY, testY, words, classes, lemmatizer, rl_epochs):
    # Build the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(len(words),), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(len(classes), activation='softmax'))

    # the rate at which the techsupport bot will learn for lets 0.50 max and min 0.10
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.25)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Implement early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    epochs = 200
    batch_size = 50  # Experiment with different batch sizes
    hist = model.fit(np.array(trainX), np.array(trainY), epochs=epochs, batch_size=batch_size, 
                     validation_data=(testX, testY), callbacks=[early_stopping], verbose=1)

    # Save the trained model
    model.save('chatbot_techsupport.h5')
    print("Training completed.")

    for epoch in range(rl_epochs):
    # Receive user query
        user_query = get_user_query()

        # Get model response
        # model_response = model.predict(process_user_query(user_query))
        model_response = model.predict(process_user_query(user_query, words, lemmatizer))


        # Receive user feedback (reward/penalty)
        reward = get_user_feedback()

        # Update model based on reward/penalty
        update_model(model_response, reward)

if __name__ == "__main__":
    trainX, testX, trainY, testY, words, classes = preprocess_data()
    lemmatizer = WordNetLemmatizer() 
    build_and_train_model(trainX, testX, trainY, testY, words, classes, lemmatizer, rl_epochs)
