/**
 * CubeSight-ML 3D Visualizer Studio Application Logic
 */

let cube3D = null;
let currentSolution = null;
let currentStep = 0;
let isPlaying = false;
let playTimer = null;
let moveSpeedMs = 300;
let latestState = null;

document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("canvas-container");
  cube3D = new RubiksCube3D(container);

  initEventListeners();
  fetchInitialState();
});

function setStatus(text, type = "normal") {
  const badge = document.getElementById("status-badge");
  const label = document.getElementById("status-text");
  label.textContent = text;
  badge.className = "status-badge";
  if (type === "working") badge.classList.add("working");
  if (type === "failed") badge.classList.add("failed");
}

async function apiCall(endpoint, method = "GET", body = null) {
  try {
    const opts = {
      method,
      headers: { "Content-Type": "application/json" },
    };
    if (body) opts.body = JSON.stringify(body);
    const resp = await fetch(endpoint, opts);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `HTTP error ${resp.status}`);
    }
    return await resp.json();
  } catch (e) {
    console.error(`API Call failed on ${endpoint}:`, e);
    setStatus(`Error: ${e.message}`, "failed");
    throw e;
  }
}

async function fetchInitialState() {
  setStatus("Connecting to Solver...", "working");
  try {
    const data = await apiCall("/api/state");
    updateUIWithState(data);
    setStatus("Ready");
  } catch (e) {
    setStatus("Connection Failed", "failed");
  }
}

function updateUIWithState(state) {
  if (!state) return;
  latestState = state;
  cube3D.updateColors(state.cubies);

  // Update stage pills & solved text
  const status = state.status || {};
  const setStagePill = (id, pillId, solved) => {
    const pill = document.getElementById(pillId);
    if (pill) pill.className = "stage-pill" + (solved ? " solved" : "");
    const lbl = document.getElementById(id);
    if (lbl) {
      lbl.textContent = solved ? "Solved" : "Not Solved";
      lbl.style.color = solved ? "#34d399" : "#a1a1aa";
    }
  };
  setStagePill("stage-status-cross", "pill-cross", status.cross_solved);
  setStagePill("stage-status-f2l", "pill-f2l", status.f2l_solved);
  setStagePill("stage-status-oll", "pill-oll", status.oll_solved);
  setStagePill("stage-status-pll", "pill-pll", status.pll_solved);

  // Update model badges & selectors
  const models = state.models || {};
  for (const s of ["cross", "f2l", "oll", "pll"]) {
    const m = models[s] || {};
    const isMock =
      m.is_mock !== undefined
        ? m.is_mock
        : m.is_heuristic !== undefined
        ? m.is_heuristic
        : true;
    const badge = document.getElementById(`model-status-${s}`);
    if (badge) {
      badge.textContent = isMock ? "Mock" : "Trained";
      badge.className = "model-badge " + (isMock ? "mock" : "trained");
      badge.title = m.checkpoint || "Distance mock model";
    }
    updateModelStageSelector(s, m);
  }

  // Update ergonomic penalties display based on active stage mode & model(s)
  updatePenaltiesDisplay();
}

function updateModelStageSelector(stage, modelInfo) {
  const select = document.getElementById(`model-select-${stage}`);
  if (!select) return;

  const currentCkpt = modelInfo.checkpoint || "";
  const avail = modelInfo.available_checkpoints || [];

  const currentOptions = Array.from(select.options).map((o) => o.value);
  const targetOptions = ["", ...avail, "__custom__"];
  const needsRebuild =
    currentOptions.length !== targetOptions.length ||
    !targetOptions.every((v, i) => currentOptions[i] === v);

  if (needsRebuild) {
    select.innerHTML = "";
    // Mock option
    const mockOpt = document.createElement("option");
    mockOpt.value = "";
    mockOpt.textContent = "Mock (No checkpoint)";
    select.appendChild(mockOpt);

    // Available checkpoints
    avail.forEach((p) => {
      const opt = document.createElement("option");
      opt.value = p;
      const parts = p.split(/[/\\]/);
      const filename = parts.pop();
      const parent = parts.pop() || "";
      opt.textContent = parent ? `${filename} (${parent})` : filename;
      select.appendChild(opt);
    });

    // Custom option
    const customOpt = document.createElement("option");
    customOpt.value = "__custom__";
    customOpt.textContent = "Custom path...";
    select.appendChild(customOpt);
  }

  const customWrap = document.getElementById(`model-custom-wrap-${stage}`);
  const customInput = document.getElementById(`model-input-${stage}`);

  if (currentCkpt === "") {
    select.value = "";
    if (customWrap) customWrap.style.display = "none";
  } else if (select.querySelector(`option[value="${CSS.escape(currentCkpt)}"]`)) {
    select.value = currentCkpt;
    if (customWrap) customWrap.style.display = "none";
  } else {
    // Checkpoint is custom
    select.value = "__custom__";
    if (customWrap) {
      customWrap.style.display = "flex";
      if (customInput && !customInput.matches(":focus")) {
        customInput.value = currentCkpt;
      }
    }
  }

  // Update penalties note on stage card
  const penNote = document.getElementById(`model-penalties-${stage}`);
  if (penNote) {
    if (modelInfo.has_penalties) {
      const penCount = modelInfo.num_penalties || 0;
      penNote.textContent = `Penalties: ${modelInfo.penalties_name} (${penCount})`;
      penNote.classList.add("active");
    } else {
      penNote.textContent = "Penalties: None (0.0)";
      penNote.classList.remove("active");
    }
  }
}

async function loadModelCheckpoint(stage, checkpoint) {
  setStatus(`Loading ${stage.toUpperCase()} model...`, "working");
  try {
    const state = await apiCall("/api/model", "POST", { stage, checkpoint });
    updateUIWithState(state);
    if (checkpoint) {
      const filename = checkpoint.split(/[/\\]/).pop();
      setStatus(`Loaded ${stage.toUpperCase()} model: ${filename}`);
    } else {
      setStatus(`Switched ${stage.toUpperCase()} to Mock model`);
    }
  } catch (err) {
    setStatus(`Failed loading ${stage.toUpperCase()} model: ${err.message}`, "failed");
  }
}

function updatePenaltiesDisplay() {
  if (!latestState) return;
  const stageSelect = document.getElementById("stage-select");
  const selectedStage = stageSelect ? stageSelect.value : "all";
  const penaltiesSelect = document.getElementById("penalties-select");
  const detail = document.getElementById("penalties-detail");

  if (!penaltiesSelect) return;

  if (selectedStage === "all") {
    const byStage = latestState.penalties?.by_stage || {};
    const stagesWithPen = Object.entries(byStage).filter(
      ([_, info]) => info.has_penalties
    );

    if (stagesWithPen.length > 0) {
      const summary = stagesWithPen
        .map(([s, info]) => `${s.toUpperCase()}: ${info.name}`)
        .join(", ");
      penaltiesSelect.innerHTML = `<option value="">Per-stage model (${summary})</option>`;
      if (detail) {
        detail.textContent = "Using each stage model's trained penalties during pipeline solve.";
      }
    } else {
      penaltiesSelect.innerHTML = '<option value="">All stages: None (zero_penalties.json)</option>';
      if (detail) {
        detail.textContent = "No move penalties trained on active models.";
      }
    }
  } else {
    const stageInfo = latestState.penalties?.by_stage?.[selectedStage] || {};
    const name = stageInfo.name || "zero_penalties.json";
    const numPen = stageInfo.num_penalties || 0;
    if (numPen > 0) {
      penaltiesSelect.innerHTML = `<option value="">${name} (${numPen} penalties)</option>`;
      if (detail) {
        detail.textContent = `Using trained penalties for ${selectedStage.toUpperCase()} model.`;
      }
    } else {
      penaltiesSelect.innerHTML = `<option value="">${name} (None / 0.0)</option>`;
      if (detail) {
        detail.textContent = `No move penalties trained for ${selectedStage.toUpperCase()} model.`;
      }
    }
  }
}

function initEventListeners() {
  // Stage select listener to update model penalties display
  const stageSelect = document.getElementById("stage-select");
  if (stageSelect) {
    stageSelect.addEventListener("change", () => {
      updatePenaltiesDisplay();
    });
  }

  // Speed slider
  const speedSlider = document.getElementById("speed-slider");
  speedSlider.addEventListener("input", (e) => {
    moveSpeedMs = parseInt(e.target.value, 10);
    document.getElementById("speed-label").textContent = `${moveSpeedMs}ms`;
  });

  // Reset button
  document.getElementById("btn-reset").addEventListener("click", async () => {
    pausePlayback();
    clearSolution();
    const state = await apiCall("/api/reset", "POST");
    updateUIWithState(state);
    setStatus("Cube Reset to Solved");
  });

  // Random Scramble
  document.getElementById("btn-scramble").addEventListener("click", async () => {
    pausePlayback();
    clearSolution();
    const depth = parseInt(document.getElementById("scramble-depth").value, 10) || 30;
    setStatus(`Scrambling (depth ${depth})...`, "working");
    const state = await apiCall("/api/scramble", "POST", { depth });
    updateUIWithState(state);
    setStatus(`Scrambled (${depth} moves)`);
  });

  // Apply custom scramble
  document.getElementById("btn-apply-custom").addEventListener("click", async () => {
    const input = document.getElementById("custom-scramble-input").value.trim();
    if (!input) return;
    pausePlayback();
    clearSolution();
    setStatus("Applying custom moves...", "working");
    const state = await apiCall("/api/scramble", "POST", { custom_moves: input });
    updateUIWithState(state);
    setStatus("Custom Scramble Applied");
  });

  // Precondition buttons
  document.querySelectorAll(".btn-precond").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const stage = e.target.dataset.stage;
      pausePlayback();
      clearSolution();
      setStatus(`Loading ${stage.toUpperCase()} Done state...`, "working");
      const state = await apiCall("/api/precondition", "POST", { stage });
      updateUIWithState(state);
      setStatus(`${stage.toUpperCase()} Done Loaded`);
    });
  });

  // Quick manual moves
  document.querySelectorAll(".move-btn").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      if (cube3D.isAnimating) return;
      const move = e.target.dataset.move;
      pausePlayback();
      try {
        const movePromise = apiCall("/api/move", "POST", { move });
        cube3D.animateMove(move, Math.min(moveSpeedMs, 200), async () => {
          const state = await movePromise;
          updateUIWithState(state);
        });
      } catch (err) {
        console.error("Manual move failed:", err);
      }
    });
  });

  // Solve button
  document.getElementById("btn-solve").addEventListener("click", async () => {
    const stage = document.getElementById("stage-select").value;
    const beamWidth = parseInt(document.getElementById("beam-width").value, 10) || 100;
    const maxSteps = parseInt(document.getElementById("max-steps").value, 10) || 40;
    const batchSize = parseInt(document.getElementById("batch-size").value, 10) || 512;

    pausePlayback();
    setStatus(`Running Beam Search (${stage.toUpperCase()})...`, "working");

    try {
      const result = await apiCall("/api/solve", "POST", {
        stage,
        beam_width: beamWidth,
        max_steps: maxSteps,
        batch_size: batchSize,
      });

      if (result.success) {
        setStatus(`Solved in ${result.time_seconds}s (${result.stm_length} STM moves)!`);
        displaySolution(result);
        startPlayback();
      } else {
        setStatus(`Search Failed after ${result.time_seconds}s`, "failed");
        clearSolution();
      }
    } catch (e) {
      setStatus(`Solve failed: ${e.message}`, "failed");
    }
  });

  // Playback navigation
  document.getElementById("btn-play-pause").addEventListener("click", togglePlayback);
  document.getElementById("btn-step-fwd").addEventListener("click", stepForward);
  document.getElementById("btn-step-bwd").addEventListener("click", stepBackward);
  document.getElementById("btn-jump-start").addEventListener("click", jumpToStart);
  document.getElementById("btn-jump-end").addEventListener("click", jumpToEnd);

  // Model checkpoint selectors
  ["cross", "f2l", "oll", "pll"].forEach((stage) => {
    const select = document.getElementById(`model-select-${stage}`);
    const customWrap = document.getElementById(`model-custom-wrap-${stage}`);
    const customInput = document.getElementById(`model-input-${stage}`);
    const applyBtn = document.querySelector(`.btn-apply-model[data-stage="${stage}"]`);

    if (select) {
      select.addEventListener("change", async (e) => {
        const val = e.target.value;
        if (val === "__custom__") {
          if (customWrap) {
            customWrap.style.display = "flex";
            if (customInput) customInput.focus();
          }
        } else {
          if (customWrap) customWrap.style.display = "none";
          await loadModelCheckpoint(stage, val || null);
        }
      });
    }

    const handleCustomSubmit = async () => {
      const path = customInput ? customInput.value.trim() : "";
      if (!path) {
        setStatus("Please enter a checkpoint path", "failed");
        return;
      }
      await loadModelCheckpoint(stage, path);
    };

    if (applyBtn) {
      applyBtn.addEventListener("click", handleCustomSubmit);
    }
    if (customInput) {
      customInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") handleCustomSubmit();
      });
    }
  });
}

function clearSolution() {
  currentSolution = null;
  currentStep = 0;
  document.getElementById("solution-chips").innerHTML = '<span style="color:#71717a;font-size:12px;">No active solution</span>';
  document.getElementById("stat-moves").textContent = "-";
  document.getElementById("stat-stm").textContent = "-";
  document.getElementById("stat-cost").textContent = "-";
  document.getElementById("stat-time").textContent = "-";
}

function displaySolution(result) {
  currentSolution = result;
  currentStep = 0;

  document.getElementById("stat-moves").textContent = result.total_length;
  document.getElementById("stat-stm").textContent = result.stm_length;
  document.getElementById("stat-cost").textContent = result.total_cost.toFixed(2);
  document.getElementById("stat-time").textContent = `${result.time_seconds}s`;

  // Render move chips
  const container = document.getElementById("solution-chips");
  container.innerHTML = "";

  result.moves.forEach((move, idx) => {
    const chip = document.createElement("button");
    chip.className = "move-chip";
    chip.id = `chip-step-${idx + 1}`;
    chip.textContent = move;
    chip.title = `Step ${idx + 1}: ${move}`;
    chip.addEventListener("click", () => jumpToStep(idx + 1));
    container.appendChild(chip);
  });

  updateChipHighlight();
}

function updateChipHighlight() {
  document.querySelectorAll(".move-chip").forEach((c, idx) => {
    if (idx + 1 === currentStep) {
      c.classList.add("current");
      c.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
    } else {
      c.classList.remove("current");
    }
  });

  // Update step label
  const total = currentSolution ? currentSolution.moves.length : 0;
  document.getElementById("step-counter").textContent = `${currentStep} / ${total}`;
}

function jumpToStep(targetStep) {
  if (!currentSolution || !currentSolution.intermediate_states) return;
  pausePlayback();

  targetStep = Math.max(0, Math.min(targetStep, currentSolution.moves.length));
  currentStep = targetStep;

  const stateData = currentSolution.intermediate_states[targetStep];
  cube3D.updateColors(stateData.cubies);
  updateChipHighlight();
}

function jumpToStart() {
  jumpToStep(0);
}

function jumpToEnd() {
  if (currentSolution) jumpToStep(currentSolution.moves.length);
}

function stepForward() {
  if (!currentSolution || currentStep >= currentSolution.moves.length) {
    pausePlayback();
    return;
  }

  const nextMove = currentSolution.moves[currentStep];
  const nextStep = currentStep + 1;

  cube3D.animateMove(nextMove, moveSpeedMs, () => {
    currentStep = nextStep;
    const stateData = currentSolution.intermediate_states[currentStep];
    cube3D.updateColors(stateData.cubies);
    updateChipHighlight();
  });
}

function stepBackward() {
  if (!currentSolution || currentStep <= 0) return;
  pausePlayback();
  jumpToStep(currentStep - 1);
}

function togglePlayback() {
  if (isPlaying) {
    pausePlayback();
  } else {
    startPlayback();
  }
}

function startPlayback() {
  if (!currentSolution || currentSolution.moves.length === 0) return;
  isPlaying = true;
  document.getElementById("btn-play-pause").innerHTML = "⏸";
  document.getElementById("btn-play-pause").classList.add("active");

  const runPlayLoop = () => {
    if (!isPlaying) return;
    if (currentStep >= currentSolution.moves.length) {
      pausePlayback();
      return;
    }
    stepForward();
    playTimer = setTimeout(runPlayLoop, moveSpeedMs + 80);
  };

  runPlayLoop();
}

function pausePlayback() {
  isPlaying = false;
  if (playTimer) clearTimeout(playTimer);
  document.getElementById("btn-play-pause").innerHTML = "▶";
  document.getElementById("btn-play-pause").classList.remove("active");
}
