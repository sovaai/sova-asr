from flask import Flask, render_template, request, send_from_directory, url_for
from file_handler import FileHandler
import json


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('speech_recognition.html')


@app.route('/asr', methods=['POST'])
def asr():
    res = []
    for f in request.files:
        if f.startswith('audio_blob') and FileHandler.check_format(request.files[f]):

            response_code, filename, response = FileHandler.get_recognized_text(request.files[f])

            if response_code == 0:
                response_audio_url = url_for('media_file', filename=filename)
            else:
                response_audio_url = None

            res.append({
                'response_audio_url': response_audio_url,
                'response_code': response_code,
                'response': response,
            })
    return json.dumps({'r': res}, ensure_ascii=False)


@app.route('/media/<path:filename>', methods=['GET'])
def media_file(filename):
    return send_from_directory('./records', filename, as_attachment=False)
