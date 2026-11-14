document.addEventListener("DOMContentLoaded", () => {
  const promptInput = document.getElementById("prompt");
  const countSelect = document.getElementById("count");
  const cfgInput = document.getElementById("cfg");
  const cfgVal = document.getElementById("cfg-val");
  const tempInput = document.getElementById("temp");
  const tempVal = document.getElementById("temp-val");
  const seedInput = document.getElementById("seed");
  const genBtn = document.getElementById("gen-btn");
  const statusLine = document.getElementById("status");
  const errorBox = document.getElementById("error-box");
  const resultsArea = document.getElementById("results-area");
  const gridAction = document.getElementById("grid-action");
  const downloadGridBtn = document.getElementById("download-grid-btn");
  const presetChips = document.querySelectorAll(".preset-chip");

  cfgInput.addEventListener("input", (e) => {
    cfgVal.textContent = parseFloat(e.target.value).toFixed(1);
  });

  tempInput.addEventListener("input", (e) => {
    tempVal.textContent = parseFloat(e.target.value).toFixed(1);
  });

  presetChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const targetPrompt = chip.getAttribute("data-prompt");
      if (targetPrompt) {
        promptInput.value = targetPrompt;
        promptInput.focus();
      }
    });
  });

  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !genBtn.disabled) {
      e.preventDefault();
      runGeneration();
    }
  });

  genBtn.addEventListener("click", () => {
    runGeneration();
  });

  async function checkServerStatus() {
    try {
      const resp = await fetch("/api/status");
      if (resp.ok) {
        const data = await resp.json();
        statusLine.textContent = `Ready (${data.device.toUpperCase()})`;
      }
    } catch (_) {
      statusLine.textContent = "Offline";
    }
  }

  checkServerStatus();

  async function runGeneration() {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      showError("Please provide a non-empty text prompt.");
      return;
    }

    clearError();
    genBtn.disabled = true;
    statusLine.textContent = "Generating 256 tokens per sprite...";

    const payload = {
      prompt: prompt,
      n: parseInt(countSelect.value, 10),
      cfg: parseFloat(cfgInput.value),
      temperature: parseFloat(tempInput.value),
      seed: seedInput.value.trim() ? parseInt(seedInput.value.trim(), 10) : null,
    };

    try {
      const resp = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || "Server error occurred during generation.");
      }

      statusLine.textContent = `Completed in ${data.elapsed_seconds.toFixed(2)}s`;

      if (data.grid_image) {
        downloadGridBtn.href = data.grid_image;
        gridAction.style.display = "block";
      } else {
        gridAction.style.display = "none";
      }

      renderSprites(data);
    } catch (err) {
      showError(err.message);
      statusLine.textContent = "Failed";
    } finally {
      genBtn.disabled = false;
    }
  }

  function renderSprites(data) {
    let html = '<div class="sprites-grid">';
    data.images.forEach((imgUri, idx) => {
      const filename = `sprite-${idx + 1}.png`;
      html += `
        <div class="sprite-card">
          <img class="sprite-img" src="${imgUri}" alt="Sprite ${idx + 1}" width="128" height="128">
          <div class="card-actions">
            <a href="${imgUri}" download="${filename}">Save PNG</a>
          </div>
        </div>
      `;
    });
    html += "</div>";

    if (data.grid_image && data.images.length > 1) {
      html += `
        <div class="sheet-section">
          <div class="sheet-title">Combined Sprite Sheet</div>
          <div class="grid-view">
            <img class="grid-img" src="${data.grid_image}" alt="Sprite sheet preview">
          </div>
        </div>
      `;
    }

    resultsArea.innerHTML = html;
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.style.display = "block";
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.style.display = "none";
  }
});
