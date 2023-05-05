# Medicine-Drugs-Recommendation-System-Using-Content-Based-Filtering 

This is a drug recommender system that uses content-based filtering to recommend drugs for a given medical condition. The system is implemented in Python using the Flask web framework, and is based on the UCI ML Drug Review dataset.

The system uses a TF-IDF vectorizer to extract features from the review text, and computes cosine similarities between reviews to identify drugs that are similar in terms of their effectiveness for a given condition. The system was trained on a subset of the dataset and tested on a separate subset, and achieved high accuracy in predicting the most effective drugs for a given condition.

The Flask app provides a user interface for entering a medical condition and viewing the top recommended drugs for that condition. The app is deployed on a cloud-based server and is available for public use.

The code for the system includes Python scripts for data preprocessing, model training, and Flask app development. Additionally, the code includes HTML templates for the app user interface.

Demo Link: http://jmanthan56.pythonanywhere.com/
