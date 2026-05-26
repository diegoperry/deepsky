const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const fileMeta = document.getElementById("fileMeta");
const statusEl = document.getElementById("status");
const resultImage = document.getElementById("resultImage");
const emptyState = document.getElementById("emptyState");
const downloadLink = document.getElementById("downloadLink");
const processButton = document.getElementById("processButton");
const dropZone = document.getElementById("dropZone");
const progressPanel = document.getElementById("progressPanel");
const progressFill = document.getElementById("progressFill");
const progressPercent = document.getElementById("progressPercent");
const progressStep = document.getElementById("progressStep");
const stepList = document.getElementById("stepList");

let currentObjectUrl = null;
let progressTimer = null;
let currentProgress = 0;

const pipelineSteps = [
  { percent: 6, label: "Upload received" },
  { percent: 18, label: "Siril FITS preparation" },
  { percent: 30, label: "OpenNGC target profile" },
  { percent: 52, label: "StarNet++ star separation" },
  { percent: 70, label: "SCUNet denoise" },
  { percent: 88, label: "OpenCV color and stretch" },
  { percent: 96, label: "Final PNG export" },
];

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
}

function clearResult() {
  stopProgress();
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
  resultImage.hidden = true;
  resultImage.removeAttribute("src");
  emptyState.hidden = false;
  downloadLink.classList.add("disabled");
  downloadLink.removeAttribute("href");
  downloadLink.setAttribute("aria-disabled", "true");
  progressPanel.hidden = true;
  progressFill.style.width = "0%";
  progressPercent.textContent = "0%";
  progressStep.textContent = "Waiting";
  [...stepList.children].forEach((item) => {
    item.classList.remove("done", "active", "failed");
  });
}

function setProgress(percent, label) {
  currentProgress = Math.max(0, Math.min(100, Math.round(percent)));
  progressPanel.hidden = false;
  progressFill.style.width = `${currentProgress}%`;
  progressPercent.textContent = `${currentProgress}%`;
  progressStep.textContent = label;

  let activeIndex = 0;
  for (let index = 0; index < pipelineSteps.length; index += 1) {
    if (currentProgress >= pipelineSteps[index].percent) activeIndex = index;
  }
  [...stepList.children].forEach((item, index) => {
    item.classList.toggle("done", index < activeIndex);
    item.classList.toggle("active", index === activeIndex && currentProgress < 100);
    item.classList.remove("failed");
  });
}

function startProgress() {
  stopProgress();
  setProgress(4, "Preparing upload");
  const startedAt = Date.now();
  progressTimer = window.setInterval(() => {
    const elapsedSeconds = (Date.now() - startedAt) / 1000;
    const target = Math.min(94, 4 + Math.log1p(elapsedSeconds) * 24);
    const nextStep = pipelineSteps.find((step) => target <= step.percent) || pipelineSteps[pipelineSteps.length - 1];
    setProgress(Math.max(currentProgress + 1, target), nextStep.label);
  }, 900);
}

function applyServerProgress(status) {
  const progress = Number.isFinite(status.progress) ? status.progress : currentProgress;
  setProgress(progress, status.step || "Running DeepSky worker pipeline");
}

async function pollJob(jobId) {
  while (true) {
    await new Promise((resolve) => window.setTimeout(resolve, 1200));
    const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new Error(error?.detail || `Job status failed with ${response.status}`);
    }
    const status = await response.json();
    applyServerProgress(status);
    if (status.status === "finished") return status;
    if (status.status === "failed") {
      throw new Error(status.error || "DeepSky worker failed.");
    }
  }
}

async function processWithQueue(formData) {
  const createResponse = await fetch("/api/jobs", {
    method: "POST",
    body: formData,
    cache: "no-store",
  });

  if (!createResponse.ok) {
    const error = await createResponse.json().catch(() => null);
    if (createResponse.status === 503 || createResponse.status === 404) {
      return processDirectly(formData);
    }
    throw new Error(error?.detail || `Job creation failed with ${createResponse.status}`);
  }

  const job = await createResponse.json();
  applyServerProgress(job);
  const finished = await pollJob(job.job_id);
  const resultResponse = await fetch(finished.result_url, { cache: "no-store" });
  if (!resultResponse.ok) {
    const error = await resultResponse.json().catch(() => null);
    throw new Error(error?.detail || `Result download failed with ${resultResponse.status}`);
  }
  return resultResponse.blob();
}

async function processDirectly(formData) {
  const response = await fetch("/api/process", {
    method: "POST",
    body: formData,
    cache: "no-store",
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail || `Processing failed with ${response.status}`);
  }
  return response.blob();
}

function completeProgress() {
  stopProgress();
  setProgress(100, "Processing complete");
  [...stepList.children].forEach((item) => {
    item.classList.add("done");
    item.classList.remove("active", "failed");
  });
}

function failProgress() {
  stopProgress();
  progressPanel.hidden = false;
  const active = [...stepList.children].find((item) => item.classList.contains("active"));
  if (active) active.classList.add("failed");
}

function stopProgress() {
  if (progressTimer) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
}

function selectedFile() {
  return fileInput.files && fileInput.files.length > 0 ? fileInput.files[0] : null;
}

fileInput.addEventListener("change", () => {
  clearResult();
  const file = selectedFile();
  fileMeta.textContent = file ? `${file.name} - ${Math.ceil(file.size / 1024)} KB` : "FITS is converted by Siril before processing";
  setStatus(file ? "Ready to process" : "No image loaded", file ? "ready" : "");
});

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");
  });
}

dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file) return;
  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  fileInput.files = dataTransfer.files;
  fileInput.dispatchEvent(new Event("change"));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearResult();

  const file = selectedFile();
  if (!file) {
    setStatus("Choose an image first.", "error");
    return;
  }

  processButton.disabled = true;
  startProgress();
  setStatus("Running the DeepSky worker pipeline...");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const blob = await processWithQueue(formData);
    currentObjectUrl = URL.createObjectURL(blob);
    resultImage.src = currentObjectUrl;
    resultImage.hidden = false;
    emptyState.hidden = true;

    downloadLink.href = currentObjectUrl;
    downloadLink.download = `${file.name.replace(/\.[^.]+$/, "")}_deepsky.png`;
    downloadLink.classList.remove("disabled");
    downloadLink.removeAttribute("aria-disabled");

    completeProgress();
    setStatus("Processed. Download now if you want to keep it.", "ready");
  } catch (error) {
    clearResult();
    failProgress();
    setStatus(error.message, "error");
  } finally {
    processButton.disabled = false;
  }
});

window.addEventListener("beforeunload", () => {
  stopProgress();
  clearResult();
});
