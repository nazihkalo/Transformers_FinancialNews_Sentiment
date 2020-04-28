import os
import sys
import logging
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np 
from serve import get_model_api
from flask_jsonpify import jsonpify
import json

# define the app
app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default


# # logging for heroku
# if 'DYNO' in os.environ:
#     app.logger.addHandler(logging.StreamHandler(sys.stdout))
#     app.logger.setLevel(logging.INFO)


# load the model
model_api = get_model_api()

# API route
@app.route('/api', methods=['POST'])
def api():
    """API function
    All model-specific logic to be defined in the get_model_api()
    function
    """
    if request.method == 'POST':
        print('POSTED SOME DATA'*10)
        input_data = request.get_json()  # Get data posted as a json
        #input_data = request.json
        app.logger.info("api_input: " + str(input_data))
        output_data = model_api(input_data)
        app.logger.info("api_output: " + str(output_data))
        df = output_data['output']
        app.logger.info("json_output: " + str(df))

        response = {'input':output_data['input'], 
                    'table': render_template('simple.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)}
    else:
        flash('That short name has already been taken. Please select another name.')
        return redirect(url_for('index'))

    return response


@app.route('/')
def index():
    return render_template('full_client.html')

# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # load the model
    model_api = get_model_api( )
    # This is used when running locally.
    app.run(host='0.0.0.0', port = int(os.environ.get("PORT", 80)), debug=True)