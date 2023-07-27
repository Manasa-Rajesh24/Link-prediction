from flask import Flask, request,render_template
import pandas as pd
import numpy as np
import model
from model import prediction_nodes


# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def main():
    

    # If a form is submitted
    if request.method == "POST":

        node1 = request.form.get("Node1")
        node2 = request.form.get("Node2")   
        if node1 == "" or node2 == "":
            prediction = "Please enter 2 nodes"
        else:
            prediction = prediction_nodes(node1,node2)    
    else:
        prediction = ""
    
    print(prediction)

    return render_template("website.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.debug=True
    app.run()





