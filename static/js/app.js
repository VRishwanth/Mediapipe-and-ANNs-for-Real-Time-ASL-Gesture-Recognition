const startBtn = document.getElementById('start-btn');
const webcam = document.getElementById('webcam');
const labelDisplay = document.getElementById('detected-letters');
const clearBtn = document.getElementById('clear-btn');
let intervalId;

startBtn.addEventListener('click', async () => {
  clearInterval(intervalId);
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.srcObject = stream;
    intervalId = setInterval(sendFrame, 500);
  } catch (e) {
    console.error('Webcam error:', e);
  }
});

function sendFrame() {
  if (!webcam.videoWidth) return;
  const canvas = document.createElement('canvas');
  canvas.width = webcam.videoWidth;
  canvas.height = webcam.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(webcam, 0, 0);
  canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append('frame', blob, 'frame.jpg');
    fetch('/predict', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(data => labelDisplay.value = data.text || 'Waiting for signs...')
      .catch(err => console.error('Predict error:', err));
  }, 'image/jpeg');
}

clearBtn.addEventListener('click', () => {
    labelDisplay.value = '';
    fetch('/clear', { method: 'POST' })
      .then(r => r.json())
      .then(data => console.log(data.status))
      .catch(err => console.error('Clear error:', err));
  });
  


