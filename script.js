let model, ortSession;

const labelsToStop = ["person", "dog", "cat"];
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const message = document.getElementById("message");

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
    audio: false,
  });
  video.srcObject = stream;
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      resolve();
    };
  });
}

async function loadModel() {
  model = await cocoSsd.load();
  ortSession = await ort.InferenceSession.create(
    "https://huggingface.co/olga-marinova/midas/resolve/main/midas.onnx"
  );
}

async function detect() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const predictions = await model.detect(video);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  let stop = false;
  let slow = false;

  predictions.forEach((pred) => {
    const [x, y, width, height] = pred.bbox;
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
    ctx.fillStyle = "lime";
    ctx.font = "16px sans-serif";
    ctx.fillText(pred.class, x, y > 10 ? y - 5 : 10);

    if (labelsToStop.includes(pred.class)) stop = true;

    if (pred.class === "traffic light") {
      const imgData = ctx.getImageData(x + width / 2, y + height / 2, 1, 1).data;
      const [r, g, b] = imgData;
      if (r > 150 && g < 100) stop = true;
      else if (r > 150 && g > 150) stop = true;
      else if (g > 150 && r < 100) slow = true;
    }
  });

  if (stop) message.textContent = "🛑 정지";
  else if (slow) message.textContent = "🟢 서행";
  else message.textContent = "감지 대기 중...";

  requestAnimationFrame(detect);
}

async function main() {
  await setupCamera();
  await loadModel();
  detect();
}

main();
