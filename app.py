from flask import Flask, render_template, request, jsonify
from jo import *  # Import all functions from jo.py
import subprocess
import sys
import os
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
csrf = CSRFProtect(app)

# Initialize the voice engine once
engine = pyttsx3.init()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/execute', methods=['POST'])
@csrf.exempt
def execute():
    try:
        data = request.json
        code = data['code']
        
        # Dynamic function mapping
        if code == "time()":
            time()
            return jsonify({'output': "Executed time command"})
        elif code == "date()":
            date()
            return jsonify({'output': "Executed date command"})
        elif code == "open_camera()":
            open_camera()
            return jsonify({'output': "Camera opened (check your desktop window)"})
        elif code == "tell_joke()":
            tell_joke()
            return jsonify({'output': "Joke told via audio"})
        else:
            return jsonify({'output': f"Unknown command: {code}"})
            
    except Exception as e:
        return jsonify({'output': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)