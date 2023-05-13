# Medicine-Drugs-Recommendation-System-Using-Content-Based-Filtering 

This system of drug recommendations uses content-based filtering to suggest medications for particular medical conditions. The Flask web framework is used to create the system, which is based on the UCI ML Drug Review dataset.

The system computes cosine similarities between reviews to find medications that are comparable in terms of their efficacy for a certain ailment. It does this by using a TF-IDF vectorizer to extract features from the review text. The algorithm showed great accuracy in predicting the most effective medications for a given ailment after being trained on a subset of the dataset and tested on a different subset.


The Flask app offers a user interface where users may enter a medical diagnosis and view the most frequently prescribed medications for that ailment. The app is accessible to the general public and is deployed on a cloud-based server.

Python scripts are used in the system's code for data preprocessing, model training, and Flask app development. The code also contains HTML templates for the app's user interface.

Demo Link: http://jmanthan56.pythonanywhere.com/
