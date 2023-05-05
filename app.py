from flask import Flask, render_template,jsonify,request
import pickle
from mdrs import recommend_drugs

with open('conditions.pickle', 'rb') as f:
    conditions = pickle.load(f)
    
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html',conditions=conditions)

@app.route('/recommend', methods=['POST'])
def recommend():
# Get the medical condition entered by the user
  condition = request.form['condition']
  # Get the top recommended drugs for the condition
  top_drugs = recommend_drugs(condition)

  top_drugs = sorted(top_drugs, key=lambda x: x['score'], reverse=True)
  top_drugs = top_drugs[:10]

  # Return the top recommended drugs as JSON
  return jsonify(top_drugs)
if __name__ == '__main__':
   app.run()



