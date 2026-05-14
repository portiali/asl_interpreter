(() => {
  const captionText = document.getElementById("caption-text");
  const chipRow = document.getElementById("chip-row");
  const transcript = document.getElementById("transcript");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const btnTranslate = document.getElementById("btn-translate");

  let currentSentence = "";
  let currentGloss = "";
  let chips = [];

  const setStatus = (kind, text) => {
    statusDot.className = `dot dot-${kind}`;
    statusText.textContent = text;
  };

  // --- Chips ---
  const clearChipsEmpty = () => {
    const empty = chipRow.querySelector(".chip-empty");
    if (empty) empty.remove();
  };
  const addChip = (word) => {
    clearChipsEmpty();
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = word.replace(/_/g, " ");
    chipRow.appendChild(chip);
    chips.push(chip);
    chipRow.scrollTop = chipRow.scrollHeight;
  };
  const fadeChips = () => chips.forEach((c) => c.classList.add("fading"));
  const resetChips = () => {
    chips.forEach((c) => c.remove());
    chips = [];
    if (!chipRow.querySelector(".chip-empty")) {
      const empty = document.createElement("span");
      empty.className = "chip-empty";
      empty.textContent = "No signs yet";
      chipRow.appendChild(empty);
    }
  };

  // --- Caption ---
  const startSentence = (gloss) => {
    currentSentence = "";
    currentGloss = gloss;
    captionText.textContent = "";
    captionText.classList.add("streaming");
    setStatus("translating", "Translating…");
  };
  const appendToken = (text) => {
    currentSentence += text;
    captionText.textContent = currentSentence;
  };
  const finishSentence = (text) => {
    captionText.classList.remove("streaming");
    captionText.textContent = text || currentSentence || "…";
    setStatus("live", "Listening");

    const entry = document.createElement("div");
    entry.className = "entry";
    const gloss = document.createElement("span");
    gloss.className = "gloss";
    gloss.textContent = `You · ${currentGloss}`;
    const body = document.createElement("span");
    body.textContent = text;
    entry.appendChild(gloss);
    entry.appendChild(body);
    transcript.appendChild(entry);
    transcript.scrollTop = transcript.scrollHeight;

    setTimeout(resetChips, 400);
  };

  // --- API ---
  const translateNow = () => fetch("/translate_now", { method: "POST" });

  // --- SSE ---
  const connect = () => {
    const es = new EventSource("/captions");
    es.onopen = () => setStatus("live", "Listening");
    es.onerror = () => setStatus("error", "Reconnecting…");
    es.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      switch (msg.type) {
        case "hello":
          setStatus("live", "Listening");
          break;
        case "sign":
          addChip(msg.word);
          break;
        case "translating":
          fadeChips();
          startSentence(msg.gloss);
          break;
        case "token":
          appendToken(msg.text);
          break;
        case "sentence_done":
          finishSentence(msg.text);
          break;
      }
    };
  };

  btnTranslate.addEventListener("click", translateNow);

  const video = document.getElementById("video");
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      const src = video.src.split("?")[0];
      video.src = `${src}?t=${Date.now()}`;
    }
  });

  connect();
})();
