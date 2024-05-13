let video;
let model;
let predictions = [];
let statusText = "Loading custom model...";
let videoReady = false;

async function preload() {
  video = createCapture(VIDEO, videoLoaded);
  video.hide();
  model = await tf.loadLayersModel('https://xyz37298.github.io/mediapipetest/model.json');
  console.log("Custom model initialized.");
  statusText = "Custom Model loaded.";
}

function videoLoaded() {
  console.log("Video initialized");
  videoReady = true;
}

function setup() {
  createCanvas(640, 480);
}

function draw() {
  background(200);
  if (model && videoReady) {
    predictHandPose();
    drawDebugView();
  }
  fill(255, 0, 0);
  textSize(32);
  text(statusText, 10, 60);
}

async function predictHandPose() {
  const prediction = await model.predict(preprocessInput(video.elt));
  updateHands(prediction);
}

function preprocessInput(imageElement) {
  // Preprocess the video input
  return tf.tidy(() => {
    return tf.browser.fromPixels(imageElement)
      .resizeNearestNeighbor([224, 224]) // Example size, adjust according to model needs
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims();
  });
}

function updateHands(predictions) {
  // Assuming predictions to be an array of detected hands
  predictions.array().then(data => {
    predictions = data; // Update the predictions
    statusText = predictions.length ? "Hand detected" : "No hands detected";
  });
}

function drawDebugView() {
  push();
  scale(0.5); // downscale the webcam capture
  image(video, 0, 0);
  pop();
}
