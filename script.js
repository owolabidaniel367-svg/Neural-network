// import NeuralNetwork from "./NeuralNetwork.js";
import NeuralNetwork from "./Ai.js";
const N = 28;
const CELL = 20;
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const output = document.getElementById("output");

const model = new NeuralNetwork(784, [32], 10);
const modelData = await (await fetch("MNIST.json")).json();
model.loadModel(modelData);

let pixels = new Array(N * N).fill(0);
let drawing = false;

function render() {
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < N * N; i++) {
    const v = pixels[i];
    if (v > 0) {
      const g = Math.round(255 * (1 - v));
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.fillRect((i % N) * CELL, Math.floor(i / N) * CELL, CELL, CELL);
    }
  }
  ctx.strokeStyle = "#ddd";
  for (let i = 0; i <= N; i++) {
    ctx.beginPath(); ctx.moveTo(i * CELL, 0); ctx.lineTo(i * CELL, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * CELL); ctx.lineTo(canvas.width, i * CELL); ctx.stroke();
  }

  const vector = new Float32Array(pixels);
  const isEmpty = pixels.every(v => v === 0);
  output.value = isEmpty ? "" : "Prediction: " + distribution(model.run(vector));
}

function paint(e) {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - rect.left) / rect.width * N);
  const y = Math.floor((e.clientY - rect.top) / rect.height * N);
  if (x < 0 || y < 0 || x >= N || y >= N) return;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= N || ny >= N) continue;
      const add = (dx === 0 && dy === 0) ? 1 : 0.3;
      const i = ny * N + nx;
      pixels[i] = Math.min(1, pixels[i] + add);
    }
  }
  render();
}

canvas.addEventListener("pointerdown", e => { drawing = true; paint(e); });
canvas.addEventListener("pointermove", e => { if (drawing) paint(e); });
window.addEventListener("pointerup", () => drawing = false);
document.getElementById("clear").addEventListener("click", () => {
  pixels.fill(0);
  render();
});

render();
function distribution(arr) {
    const sum = arr.reduce((a, b) => a + b, 0);
    return Array.from(arr, (v, i) => ({ digit: i, percent: (v / sum) * 100 }))
        .sort((a, b) => b.percent - a.percent)
        .map(d => `${d.digit}: ${d.percent.toFixed(0)}%`)
        .join("\n");
}