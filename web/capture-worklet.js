// Collects the audio flowing through the graph into 50 ms mono blocks and posts them to the page.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = Math.round(sampleRate * 0.05);
    this.buf = new Float32Array(this.size);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    let i = 0;
    while (i < ch.length) {
      const k = Math.min(ch.length - i, this.size - this.n);
      this.buf.set(ch.subarray(i, i + k), this.n);
      this.n += k;
      i += k;
      if (this.n === this.size) {
        this.port.postMessage(this.buf, [this.buf.buffer]);
        this.buf = new Float32Array(this.size);
        this.n = 0;
      }
    }
    return true;
  }
}

registerProcessor("capture", Capture);
