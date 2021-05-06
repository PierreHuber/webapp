# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_flex_quickstart]
import logging

from flask import Flask, render_template, request


app = Flask(__name__)
from google.cloud import automl
import os

#JSON key files:
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "bsa-2021-project-4cf65702700c.json"

def predictor(content):
    """Predict."""
    # [START automl_language_text_classification_predict]
    prediction_client = automl.PredictionServiceClient()
    project_id = "bsa-2021-project"
    model_id = "TCN1057987471641411584"

    # Get the full path of the model.
    model_full_id = automl.AutoMlClient.model_path(project_id, "us-central1", model_id)

    # Supported mime_types: 'text/plain', 'text/html'
    # https://cloud.google.com/automl/docs/reference/rpc/google.cloud.automl.v1#textsnippet
    text_snippet = automl.TextSnippet(content=content, mime_type="text/plain")
    payload = automl.ExamplePayload(text_snippet=text_snippet)

    response = prediction_client.predict(name=model_full_id, payload=payload)
    magic = (u"Predicted class name: {}".format(response.payload[0].display_name) + " " + u"with a predicted class score of {}".format(response.payload[0].classification.score))
    return magic




    # [END automl_language_text_classification_predict]



@app.route('/')
def home():
	return render_template('interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	my_prediction = predictor(request.form['message'])
    	return render_template('result.html',message = request.form['message'] , prediction = str(my_prediction))


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
	app.run(debug = True)
# [END gae_flex_quickstart]
