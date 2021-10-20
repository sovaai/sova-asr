import os
import subprocess
import time
import logging
import uuid
from speech_recognizer import SpeechRecognizer
from punctuator import Punctuator
from number_utils.text2numbers import TextToNumbers


speech_recognizer = SpeechRecognizer()
punctuator = Punctuator(model_path="data/punctuator")
text2numbers = TextToNumbers()


class FileHandler:
    @staticmethod
    def get_recognized_text(blob):
        try:
            filename = str(uuid.uuid4())
            os.makedirs('./records', exist_ok=True)
            new_record_path = os.path.join('./records', filename + '.webm')
            blob.save(new_record_path)
            new_filename = filename + '.wav'
            converted_record_path = FileHandler.convert_to_wav(new_record_path, new_filename)
            response_models_result = FileHandler.get_models_result(converted_record_path)
            return 0, new_filename, response_models_result
        except Exception as e:
            logging.exception(e)
            return 1,  None, str(e)

    @staticmethod
    def convert_to_wav(webm_full_filepath, new_filename):
        converted_record_path = os.path.join('./records', new_filename)
        subprocess.call('ffmpeg -i {0} -ar 16000 -b:a 256k -ac 1 -sample_fmt s16 {1}'.format(
                webm_full_filepath, converted_record_path
            ),
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        os.remove(webm_full_filepath)
        return converted_record_path

    @staticmethod
    def check_format(files):
        return (files.mimetype.startswith('audio/') or [
            files.filename.endswith(audio_format) for audio_format in [
                'mp3', 'ogg', 'acc', 'flac', 'au', 'm4a', 'mp4', 'mov', 'avi', 'wmv', '3gp', 'flv', 'mkv'
            ]
        ])
        return True

    @staticmethod
    def get_models_result(converted_record_path, delimiter='<br>'):
        results = []
        start = time.time()
        decoder_result = speech_recognizer.recognize(converted_record_path)
        text = punctuator.predict(decoder_result.text)
        text = text2numbers.convert(text)
        end = time.time()
        results.append(
            {
                'text': text,
                'time': round(end - start, 3),
                'confidence': decoder_result.score,
                'words': decoder_result.words
            }
        )
        return results
