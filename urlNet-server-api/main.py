import requests
import json
from types import SimpleNamespace
from urlnet import *
from flask import Flask
from flask_cors import CORS
from flask_restful import Api, Resource, reqparse
from util import exit_on_busy_port

app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={
    r"/*": {
        "origins": "*"
    }
})

@app.route('/check-url/<string:url>', methods=['GET'])
def check_url(url=""):
    if(url == ""):
        return { "error_message": "URL is empty" }, 404
    else:
        res, softmax_score = main_logic(url)
        return { "is_phishing": res, "softmax_score": str(softmax_score) }, 200

@app.route('/check-url-vt/<string:url>', methods=['GET'])
def check_url_vt(url=""):
    if(url == ""):
        return { "error_message": "URL is empty" }, 404
    else:
        headers = { 'x-apikey': '<api_key>' }
        responseDict = requests.get("https://www.virustotal.com/api/v3/domains/{}".format(url), headers=headers).json()

        for key in responseDict["data"]["attributes"]["last_analysis_results"]:
            if(responseDict["data"]["attributes"]["last_analysis_results"][key]["category"] == "malicious"):
                return { "vt_confirmed": True }, 200
                
        return { "vt_confirmed": False }, 200

        


exit_on_busy_port()
app.run(debug=True)