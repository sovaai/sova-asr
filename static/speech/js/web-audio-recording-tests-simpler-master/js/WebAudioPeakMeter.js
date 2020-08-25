/**
 * Copied from https://github.com/esonderegger/web-audio-peak-meter
 * Modified to class form to allow multiple instances on a page.
 */
class WebAudioPeakMeter {
  constructor () {
    this.options = {
      borderSize: 2,
      fontSize: 9,
      backgroundColor: 'white',
      tickColor: '#000',
      gradient: ['red 1%', '#ff0 16%', 'lime 45%', '#080 100%'],
      dbRange: 48,
      dbTickSize: 6,
      maskTransition: 'height 0.1s'
    }

    this.vertical = true
    this.channelCount = 1
    this.channelMasks = []
    this.channelPeaks = []
    this.channelPeakLabels = []
  }

  getBaseLog (x, y) {
    return Math.log(y) / Math.log(x)
  }

  dbFromFloat (floatVal) {
    return this.getBaseLog(10, floatVal) * 20
  }

  setOptions (userOptions) {
    for (var k in userOptions) {
      this.options[k] = userOptions[k]
    }
    this.tickWidth = this.options.fontSize * 2.0
    this.meterTop = this.options.fontSize * 1.5 + this.options.borderSize
  }

  createMeterNode (sourceNode, audioCtx) {
    var c = sourceNode.channelCount
    var meterNode = audioCtx.createScriptProcessor(2048, c, c)
    sourceNode.connect(meterNode)
    meterNode.connect(audioCtx.destination)
    return meterNode
  }

  createContainerDiv (parent) {
    var meterElement = document.createElement('div')
    meterElement.style.position = 'relative'
    meterElement.style.width = this.elementWidth + 'px'
    meterElement.style.height = this.elementHeight + 'px'
    meterElement.style.backgroundColor = this.options.backgroundColor
    parent.appendChild(meterElement)
    return meterElement
  }

  createMeter (domElement, meterNode, optionsOverrides) {
    this.setOptions(optionsOverrides)
    this.elementWidth = domElement.clientWidth
    this.elementHeight = domElement.clientHeight
    var meterElement = this.createContainerDiv(domElement)
    if (this.elementWidth > this.elementHeight) {
      this.vertical = false
    }
    this.meterHeight = this.elementHeight - this.meterTop - this.options.borderSize
    this.meterWidth = this.elementWidth - this.tickWidth - this.options.borderSize
    this.createTicks(meterElement)
    this.createRainbow(meterElement, this.meterWidth, this.meterHeight,
      this.meterTop, this.tickWidth)
    this.channelCount = meterNode.channelCount
    var channelWidth = this.meterWidth / this.channelCount
    var channelLeft = this.tickWidth
    for (var i = 0; i < this.channelCount; i++) {
      this.createChannelMask(meterElement, this.options.borderSize,
        this.meterTop, channelLeft, false)
      this.channelMasks[i] = this.createChannelMask(meterElement, channelWidth,
        this.meterTop, channelLeft,
        this.options.maskTransition)
      this.channelPeaks[i] = 0.0
      this.channelPeakLabels[i] = this.createPeakLabel(meterElement, channelWidth,
        channelLeft)
      channelLeft += channelWidth
    }
    meterNode.onaudioprocess = (e) => this.updateMeter(e)
    meterElement.addEventListener('click', function () {
      for (var i = 0; i < this.channelCount; i++) {
        this.channelPeaks[i] = 0.0
        this.channelPeakLabels[i].textContent = '-∞'
      }
    }, false)
  }

  createTicks (parent) {
    var numTicks = Math.floor(this.options.dbRange / this.options.dbTickSize)
    var dbTickLabel = 0
    var dbTickTop = this.options.fontSize + this.options.borderSize
    for (var i = 0; i < numTicks; i++) {
      var dbTick = document.createElement('div')
      parent.appendChild(dbTick)
      dbTick.style.width = this.tickWidth + 'px'
      dbTick.style.textAlign = 'right'
      dbTick.style.color = this.options.tickColor
      dbTick.style.fontSize = this.options.fontSize + 'px'
      dbTick.style.position = 'absolute'
      dbTick.style.top = dbTickTop + 'px'
      dbTick.textContent = dbTickLabel + ''
      dbTickLabel -= this.options.dbTickSize
      dbTickTop += this.meterHeight / numTicks
    }
  }

  createRainbow (parent, width, height, top, left) {
    var rainbow = document.createElement('div')
    parent.appendChild(rainbow)
    rainbow.style.width = width + 'px'
    rainbow.style.height = height + 'px'
    rainbow.style.position = 'absolute'
    rainbow.style.top = top + 'px'
    rainbow.style.left = left + 'px'
    var gradientStyle = 'linear-gradient(' + this.options.gradient.join(', ') + ')'
    rainbow.style.backgroundImage = gradientStyle
    return rainbow
  }

  createPeakLabel (parent, width, left) {
    var label = document.createElement('div')
    parent.appendChild(label)
    label.style.width = width + 'px'
    label.style.textAlign = 'center'
    label.style.color = this.options.tickColor
    label.style.fontSize = this.options.fontSize + 'px'
    label.style.position = 'absolute'
    label.style.top = this.options.borderSize + 'px'
    label.style.left = left + 'px'
    label.textContent = '-∞'
    return label
  }

  createChannelMask (parent, width, top, left, transition) {
    var channelMask = document.createElement('div')
    parent.appendChild(channelMask)
    channelMask.style.width = width + 'px'
    channelMask.style.height = this.meterHeight + 'px'
    channelMask.style.position = 'absolute'
    channelMask.style.top = top + 'px'
    channelMask.style.left = left + 'px'
    channelMask.style.backgroundColor = this.options.backgroundColor
    if (transition) {
      channelMask.style.transition = this.options.maskTransition
    }
    return channelMask
  }

  maskSize (floatVal) {
    if (floatVal === 0.0) {
      return this.meterHeight
    }
    else {
      var d = this.options.dbRange * -1
      var returnVal = Math.floor(this.dbFromFloat(floatVal) * this.meterHeight / d)
      if (returnVal > this.meterHeight) {
        return this.meterHeight
      }
      else {
        return returnVal
      }
    }
  }

  updateMeter (audioProcessingEvent) {
    var inputBuffer = audioProcessingEvent.inputBuffer
    var i
    var channelData = []
    var channelMaxes = []
    for (i = 0; i < this.channelCount; i++) {
      channelData[i] = inputBuffer.getChannelData(i)
      channelMaxes[i] = 0.0
    }
    for (var sample = 0; sample < inputBuffer.length; sample++) {
      for (i = 0; i < this.channelCount; i++) {
        if (Math.abs(channelData[i][sample]) > channelMaxes[i]) {
          channelMaxes[i] = Math.abs(channelData[i][sample])
        }
      }
    }
    for (i = 0; i < this.channelCount; i++) {
      var thisMaskSize = this.maskSize(channelMaxes[i], this.meterHeight)
      this.channelMasks[i].style.height = thisMaskSize + 'px'
      if (channelMaxes[i] > this.channelPeaks[i]) {
        this.channelPeaks[i] = channelMaxes[i]
        var labelText = this.dbFromFloat(this.channelPeaks[i]).toFixed(1)
        this.channelPeakLabels[i].textContent = labelText
      }
    }
  }
}
