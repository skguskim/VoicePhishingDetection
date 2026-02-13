class RecorderWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.sampleRate = sampleRate;
    this.maxSamples = this.sampleRate * 5; // 5초
    this.count = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channel = input[0];
    this.buffer.push(new Float32Array(channel));
    this.count += channel.length;

    if (this.count >= this.maxSamples) {
      const pcm = new Float32Array(this.count);
      let offset = 0;
      for (const chunk of this.buffer) {
        pcm.set(chunk, offset);
        offset += chunk.length;
      }

      this.port.postMessage(pcm);
      this.buffer = [];
      this.count = 0;
    }
    return true;
  }
}

registerProcessor("recoder-worklet", RecorderWorklet);