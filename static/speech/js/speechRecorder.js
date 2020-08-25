String.prototype.format = function () {
    a = this;
    for (k in arguments) {
        a = a.replace("{" + k + "}", arguments[k]);
    }
    return a;
};

$(document).ready(function () {
    $("#multiaudio").change(function () {
        MULTIFILE = true;
        RECORDING = 2;
        send_records();
    });
});

const send_records = (blob) => {
    ERROR_FIELD = document.getElementById("error_field");
    RECOGNITION_RESULT = document.getElementById("table_body_result");

    var data = new FormData();

    if (MULTIFILE) {
        var blobs = $("#multiaudio")[0].files;
        for (var i = 0; i < blobs.length; i++) {
            data.append("audio_blob_" + i, blobs[i]);
        }
        MULTIFILE = false;
    } else {
        data.append("audio_blob", blob ? blob : BLOB);
    }

    fetch("/asr/", { method: "post", body: data })
        .then((response) => {
            if (!response.ok) throw response;
            return response.json();
        })
        .then((response) => {
            RECOGNITION_RESULT.innerHTML = "";
            for (i of response["r"]) {
                response_code = i["response_code"];
                response_audio_url = i["response_audio_url"];
                response = i["response"];

                if (response_code === 0) {
                    response.forEach(function (model_ans) {
                        RECOGNITION_RESULT.insertAdjacentHTML(
                            "beforeend",
                            TR_PATTERN.format(
                                response_audio_url,
                                model_ans["time"],
                                model_ans["text"]
                            )
                        );
                    });
                } else {
                    ERROR_FIELD.innerHTML = "Error: " + response;
                    ERROR_FIELD.style.display = "block";
                }
            }
        })
        .catch((err) => {
            console.log(err);
            err.text().then((errorMessage) => {
                ERROR_FIELD.innerHTML = errorMessage;
                ERROR_FIELD.style.display = "block";
            });
        });
};

const recordAudio = (stream) =>
    new Promise(async (resolve) => {
        const mediaRecorder = new MediaRecorder(stream);
        var audioChunks = [];

        mediaRecorder.addEventListener("dataavailable", (event) => {
            audioChunks.push(event.data);
        });

        const start = () => {
            STOP = true;
            audioChunks = [];
            mediaRecorder.start();
        };

        const stop = () =>
            new Promise((resolve) => {
                mediaRecorder.addEventListener("stop", () => {
                    STOP = false;
                    BLOB = new Blob(audioChunks);
                    send_records();
                });

                try {
                    mediaRecorder.stop();
                } finally {
                }
            });
        resolve({ start, stop });
    });

const startRecord = async () => {
    REC_BTN = document.getElementById("rec_btn");

    if (STOP) {
        REC_BTN.className = "fa fa-microphone act_btn";
        await recorder.stop();
        return;
    }
    recorder = await recordAudio(stream);
    recorder.start();
    REC_BTN.className = "fa fa-pause exe_btn";
};

document.addEventListener("DOMContentLoaded", () => {
    navigator.getUserMedia =
        navigator.mediaDevices.getUserMedia ||
        navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia ||
        navigator.msGetUserMedia;
    navigator.mediaDevices
        .getUserMedia({
            audio: { sampleSize: 24, channelCount: 1, sampleRate: 44100 },
        })
        .then((r) => (stream = r));
});

var recorder,
    stream,
    BLOB = {};
var MULTIFILE = false,
    STOP = false;
var TR_PATTERN = `
    <tr>
        <td>
            <audio controls preload="none">
                <source src="{0}" type="audio/wav">
                Your browser does not support audio recording.
            </audio>
        </td>
        <td>{1}</td>
        <td>{2}</td>
    </tr>
`;
