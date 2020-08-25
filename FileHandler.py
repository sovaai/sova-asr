import os
import subprocess
import time
import logging
import uuid
from SpeechRecognizer import SpeechRecognizer


speechRecognizer = SpeechRecognizer()


class FileHandler:
    @staticmethod
    def get_recognized_text(blob):
        try:
            filename = str(uuid.uuid4())
            os.makedirs('./Records', exist_ok=True)
            new_record_path = os.path.join('./Records', filename + '.webm')
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
        converted_record_path = os.path.join('./Records', new_filename)
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
        return (files.mimetype.startswith('audio/') or
                [files.filename.endswith(audio_format) for audio_format in ['mp3', 'ogg', 'acc', 'flac', 'au', 'm4a']])

    @staticmethod
    def get_models_result(converted_record_path, delimiter='<br>'):
        results = []
        start = time.time()
        recognized_texts = speechRecognizer.recognize(converted_record_path)
        end = time.time()
        results.append(
            {
                'text': delimiter.join([str(text) for text in recognized_texts if text is not None]),
                'time': round(end - start, 3)
            }
        )
        return results
