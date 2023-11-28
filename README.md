Chatbot TechSupport

This project implements a simple chatbot with a focus on providing technical support responses. The chatbot is trained using a neural network and natural language processing techniques to understand and respond to user queries.

Prerequisites
Make sure you have the following dependencies installed:

Python 3
TensorFlow
Keras
NLTK (Natural Language Toolkit)
Install the required Python packages using the following command:

bash
Copy code
pip install tensorflow keras nltk
Project Structure
intents.json: Contains predefined intents and responses for training the chatbot.
words.pkl and classes.pkl: Pickle files storing preprocessed words and classes.
chatbot_techsupport.h5: Trained neural network model saved in HDF5 format.
chatbot_techsupport.py: Python script containing the chatbot implementation.
How to Run
Clone the repository:
bash
Copy code
git clone https://github.com/karabocodes/TechSupportBot
cd chatbot-techsupport
Run the chatbot script:
bash
Copy code
python chatbot_techsupport.py
Interact with the chatbot by entering messages. The chatbot will predict the intent of the user's message and provide a relevant response.
Additional Information
The training data and responses are defined in the intents.json file. You can customize this file to add new intents or modify existing ones.

The neural network model is trained using the chatbot_techsupport.h5 file. If you want to retrain the model, you can modify the training code in the script and run it.

The chatbot uses a bag-of-words model for message processing and prediction. You can explore more advanced techniques for better performance.
