"use strict";

class App {
  constructor () {
    this.recBtn = document.getElementById('rec_btn');

    this.isRecording = false
    this.saveNextRecording = false

    this.stop = false
  }

  init () {
    this._initEventListeners()
  }

  _initEventListeners () {

    this.recBtn.addEventListener('click', evt => {
      if (this.stop) {

        this.stop = false;

        this._stopAllRecording();
        this.recBtn.className = 'fa fa-microphone act_btn';

      } else {
        
        this.stop = true

        this._stopAllRecording()
        this.saveNextRecording = true
        this._startRecording()

        this.recBtn.className = "fa fa-pause exe_btn";
      }
    })
  }

  _startRecording () {
    if (!this.recorderSrvc) {
      this.recorderSrvc = new RecorderService()
      this.recorderSrvc.em.addEventListener('recording', (evt) => this._onNewRecording(evt))
    }

    if (!this.webAudioPeakMeter) {
      this.webAudioPeakMeter = new WebAudioPeakMeter()
      this.meterEl = document.getElementById('recording-meter')
    }

    this.recorderSrvc.onGraphSetupWithInputStream = (inputStreamNode) => {
      this.meterNodeRaw = this.webAudioPeakMeter.createMeterNode(inputStreamNode, this.recorderSrvc.audioCtx)
      this.webAudioPeakMeter.createMeter(this.meterEl, this.meterNodeRaw, {})
    }

    this.recorderSrvc.startRecording()
    this.isRecording = true
  }

  _stopAllRecording () {
    if (this.recorderSrvc && this.isRecording) {

      this.recorderSrvc.stopRecording()
      this.isRecording = false

      if (this.meterNodeRaw) {
        this.meterNodeRaw.disconnect()
        this.meterNodeRaw = null
        this.meterEl.innerHTML = ''
      }
    }
  }

  _onNewRecording (evt) {
    if (!this.saveNextRecording) {
      return
    }
    send_records(evt.detail.recording.blob)
  }
}
