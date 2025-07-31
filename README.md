# Sentiment Classifier in Tweets using a Neural Network

This project is part of a shared course assignment for **Foundations of Language Technology**, completed during the **Winter 2021/2022** term. The goal of the project was to design and train a simple neural network model to detect **sentiments** in tweets and classify them into positive, negative, or other categories.

## Overview

Social media platforms like Twitter are rich in textual data, making them ideal for sentiment analysis tasks. In this project, we implemented a basic feedforward neural network to classify tweets into three classes:

* `0`: Positive sentiment
* `1`: Negative sentiment
* `2`: Other (neutral or unassigned)

The model was trained using preprocessed tweet data and leverages fundamental natural language processing techniques.

## Technologies Used

* **TensorFlow**: Used to build and train the neural network.
* **NumPy**: Used for numerical operations, data manipulation, and preprocessing.
* **JSON**: Used for loading and handling structured tweet data.
* **NLTK**: Used for stopword removal, lemmatization, and stemming.
* **re**, **string**: Used for filtering links and punctuation.

## Project Structure

```plaintext
sentiment classifier/
│
├── data/
│   └── train.csv           # Training dataset containing tweets and their sentiment labels
│   └── trial.csv           # Trial dataset containing tweets without their sentiment labels   
|   └── test.csv            # Test dataset containing tweets and their sentiment labels

│
├── models/
│   └── model.h5            # Trained neural network model
|   └── model.json          # Trained neural network model
|   └── BoW.txt             # Bag-of-Words representation of tweets used for model input and inspection
│
├── src/
│   ├── preprocess.py         # Code for cleaning and vectorizing tweet data
│   ├── model.py              # Code defining and training the neural network
│   └── inference.py          # Code for running the model
│
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Data Format

The dataset is provided as a CSV file with the following structure:

```csv
date,country,lang,content,label
2021-10-23 16:20:41+00:00,Germany,de,"Plakataktion zur Thematik „Muslime gegen Rasissmus“ von 22.10.2021 - 01.11.2021 

#Islam #Muslimegegenrassismus #Ahmadiyya #Menschenrechte https://t.co/gEVGGN0GSL",0
2015-05-16 05:37:36+00:00,France,en,"Worth every penny to come here! Even Just for the beach. 
#SubhanAllah 🙏😍💖😇

From the top view of… https://t.co/0RqLsnlorJ",2
```



### Note on Class Imbalance

During evaluation, it was observed that the model disproportionately predicted class `2` ("other") due to a significant class imbalance in the dataset. The dominance of this class likely caused the model to converge toward always selecting it. Potential solutions include:

* Further subdividing class `2` to reduce its generality.
* Augmenting the dataset with more examples of class `0` (positive) and class `1` (negative) tweets.
* Applying class weighting or resampling techniques during training to address imbalance.

## Future Improvements

* Integrate pre-trained embeddings (e.g., GloVe, Word2Vec)
* Experiment with recurrent (LSTM/GRU) or transformer-based models
* Add class weighting or data augmentation for imbalanced classes
* Extend to sentiment intensity regression or multi-party analysis

## Acknowledgements

This project was completed as part of the **Foundations of Language Technology** course in Winter 2021/2022. Special thanks to the course instructors and all collaborators.

---

© 2022 - Course Project Submission
