class RecorderService {
  constructor () {
    window.AudioContext = window.AudioContext || window.webkitAudioContext

    this.em = document.createDocumentFragment()

    this.state = 'inactive'

    this.chunks = []
    this.chunkType = ''

    this.encoderMimeType = 'audio/wav'

    this.config = {
      manualEncoderId: 'wav',
      micGain: 1.0,
      processorBufferSize: 2048,
      stopTracksAndCloseCtxWhenFinished: true,
      usingMediaRecorder: typeof window.MediaRecorder !== 'undefined',
      userMediaConstraints: { audio: { echoCancellation: false } }
    }
  }

  /* Returns promise */
  startRecording () {
    if (this.state !== 'inactive') {
      return
    }

    // This is the case on ios/chrome, when clicking links from within ios/slack (sometimes), etc.
    if (!navigator || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Missing support for navigator.mediaDevices.getUserMedia') // temp: helps when testing for strange issues on ios/safari
      return
    }

    this.audioCtx = new AudioContext()
    this.micGainNode = this.audioCtx.createGain()
    this.outputGainNode = this.audioCtx.createGain()

    // If not using MediaRecorder(i.e. safari and edge), then a script processor is required. It's optional
    // on browsers using MediaRecorder and is only useful if you want to do custom analysis or manipulation of
    // recorded audio data.
    if (!this.config.usingMediaRecorder) {
      this.processorNode = this.audioCtx.createScriptProcessor(this.config.processorBufferSize, 1, 1) // TODO: Get the number of channels from mic
    }

    // Create stream destination on chrome/firefox because, AFAICT, we have no other way of feeding audio graph output
    // in to MediaRecorder. Safari/Edge don't have this method as of 2018-04.
    if (this.audioCtx.createMediaStreamDestination) {
      this.destinationNode = this.audioCtx.createMediaStreamDestination()
    }
    else {
      this.destinationNode = this.audioCtx.destination
    }

    // Create web worker for doing the encoding
    if (!this.config.usingMediaRecorder) {
      this.encoderWorker = new Worker('static/speech/js/web-audio-recording-tests-simpler-master/js/encoder-wav-worker.js')
      this.encoderMimeType = 'audio/wav'

      this.encoderWorker.addEventListener('message', (e) => {
        let event = new Event('dataavailable')
        event.data = new Blob(e.data, { type: this.encoderMimeType })
        this._onDataAvailable(event)
      })
    }

    // This will prompt user for permission if needed
    return navigator.mediaDevices.getUserMedia(this.config.userMediaConstraints)
      .then((stream) => {
        this._startRecordingWithStream(stream)
      })
      .catch((error) => {
        alert('Error with getUserMedia: ' + error.message) // temp: helps when testing for strange issues on ios/safari
        console.log(error)
      })
  }

  _startRecordingWithStream (stream) {
    this.micAudioStream = stream

    this.inputStreamNode = this.audioCtx.createMediaStreamSource(this.micAudioStream)
    this.audioCtx = this.inputStreamNode.context

    // Allow optionally hooking in to audioGraph inputStreamNode, useful for meters
    if (this.onGraphSetupWithInputStream) {
      this.onGraphSetupWithInputStream(this.inputStreamNode)
    }

    this.inputStreamNode.connect(this.micGainNode)
    this.micGainNode.gain.setValueAtTime(this.config.micGain, this.audioCtx.currentTime)

    let nextNode = this.micGainNode

    this.state = 'recording'

    if (this.processorNode) {
      nextNode.connect(this.processorNode)
      this.processorNode.connect(this.outputGainNode)
      this.processorNode.onaudioprocess = (e) => this._onAudioProcess(e)
    }
    else {
      nextNode.connect(this.outputGainNode)
    }

    this.outputGainNode.connect(this.destinationNode)

    if (this.config.usingMediaRecorder) {
      this.mediaRecorder = new MediaRecorder(this.destinationNode.stream)
      this.mediaRecorder.addEventListener('dataavailable', (evt) => this._onDataAvailable(evt))
      this.mediaRecorder.addEventListener('error', (evt) => this._onError(evt))

      this.mediaRecorder.start()
    }
    else {
      // Output gain to zero to prevent feedback. Seems to matter only on Edge, though seems like should matter
      // on iOS too.  Matters on chrome when connecting graph to directly to audioCtx.destination, but we are
      // not able to do that when using MediaRecorder.
      this.outputGainNode.gain.setValueAtTime(0, this.audioCtx.currentTime)
    }
  }

  _onAudioProcess (e) {
    if (this.config.broadcastAudioProcessEvents) {
      this.em.dispatchEvent(new CustomEvent('onaudioprocess', {
        detail: {
          inputBuffer: e.inputBuffer,
          outputBuffer: e.outputBuffer
        }
      }))
    }

    // Safari and Edge require manual encoding via web worker. Single channel only for now.
    // Example stereo encoderWav: https://github.com/MicrosoftEdge/Demos/blob/master/microphone/scripts/recorderworker.js
    if (!this.config.usingMediaRecorder) {
      if (this.state === 'recording') {
        if (this.config.broadcastAudioProcessEvents) {
          this.encoderWorker.postMessage(['encode', e.outputBuffer.getChannelData(0)])
        }
        else {
          this.encoderWorker.postMessage(['encode', e.inputBuffer.getChannelData(0)])
        }
      }
    }
  }

  stopRecording () {
    if (this.state === 'inactive') {
      return
    }

    if (this.config.usingMediaRecorder) {
      this.state = 'inactive'
      this.mediaRecorder.stop()
    }
    else {
      this.state = 'inactive'
      this.encoderWorker.postMessage(['dump', this.audioCtx.sampleRate])
    }
  }

  _onDataAvailable (evt) {
    this.chunks.push(evt.data)
    this.chunkType = evt.data.type

    if (this.state !== 'inactive') {
      return
    }

    let blob = new Blob(this.chunks, { 'type': this.chunkType })
    let blobUrl = URL.createObjectURL(blob)
    const recording = {
      ts: new Date().getTime(),
      blobUrl: blobUrl,
      mimeType: blob.type,
      size: blob.size,
      blob: blob
    }

    this.chunks = []
    this.chunkType = null

    if (this.destinationNode) {
      this.destinationNode.disconnect()
      this.destinationNode = null
    }
    if (this.outputGainNode) {
      this.outputGainNode.disconnect()
      this.outputGainNode = null
    }

    if (this.processorNode) {
      this.processorNode.disconnect()
      this.processorNode = null
    }

    if (this.encoderWorker) {
      this.encoderWorker.postMessage(['close'])
      this.encoderWorker = null
    }

    if (this.micGainNode) {
      this.micGainNode.disconnect()
      this.micGainNode = null
    }
    if (this.inputStreamNode) {
      this.inputStreamNode.disconnect()
      this.inputStreamNode = null
    }

    if (this.config.stopTracksAndCloseCtxWhenFinished) {
      // This removes the red bar in iOS/Safari
      this.micAudioStream.getTracks().forEach((track) => track.stop())
      this.micAudioStream = null

      this.audioCtx.close()
      this.audioCtx = null
    }

    this.em.dispatchEvent(new CustomEvent('recording', { detail: { recording: recording } }))
  }

  _onError (evt) {
    console.log('error', evt)
    this.em.dispatchEvent(new Event('error'))
    alert('error:' + evt) // for debugging purposes
  }
}
