"use strict";

(() => {
  const box = document.querySelector(".steering-demo");
  if (!box) return;
  // Only this data changes when recorded benchmark examples are supplied.
  // These are illustrative placeholders, not measured model responses.
  let example = {
    question: "How are you doing?",
    baseline: "I'm doing well, thank you. How can I help you today?",
    steered: "我很好，谢谢。今天有什么可以帮你的吗？",
  };
  const question = box.querySelector(".steering-demo-question-text");
  const response = box.querySelector(".steering-demo-answer-text");
  const description = box.querySelector(".steering-demo-description");
  const pauseButton = box.querySelector(".steering-demo-pause");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const glyphs = Array.from("▓▒░<>/\\#文言語字");
  let questionChars, baselineChars, targetChars, responseChars, timing;
  let elapsed = 0;
  let previousTime = null;
  let frame = null;
  let inView = false;
  let paused = false;
  let started = false;

  function characterNodes(node, count) {
    const nodes = Array.from({ length: count }, () => {
      const span = document.createElement("span");
      span.className = "steering-demo-char";
      return span;
    });
    node.replaceChildren(...nodes);
    return nodes;
  }

  function prepare() {
    const q = Array.from(example.question);
    baselineChars = Array.from(example.baseline);
    targetChars = Array.from(example.steered);
    questionChars = characterNodes(question, q.length);
    questionChars.forEach((span, i) => { span.textContent = q[i]; });
    responseChars = characterNodes(response, Math.max(baselineChars.length, targetChars.length));
    const questionEnd = 400 + q.length * 55;
    const baselineStart = questionEnd + 550;
    const baselineEnd = baselineStart + baselineChars.length * 42;
    const steeringStart = baselineEnd + 1100;
    const steeringEnd = steeringStart + 2400;
    timing = { questionEnd, baselineStart, baselineEnd, steeringStart, steeringEnd, end: steeringEnd + 4600 };
    elapsed = 0;
    previousTime = null;
    render();
  }

  function phaseAt(time) {
    if (time < 400) return "waiting";
    if (time < timing.baselineStart) return "question";
    if (time < timing.baselineEnd) return "baseline";
    if (time < timing.steeringStart) return "baseline-hold";
    if (time < timing.steeringEnd) return "steering";
    return "steered";
  }

  function render() {
    const phase = phaseAt(elapsed);
    if (box.dataset.phase !== phase) {
      box.dataset.phase = phase;
      if (phase === "baseline-hold") description.textContent = `Question: ${example.question} Original response: ${example.baseline}`;
      if (phase === "steered") description.textContent = `Steered response: ${example.steered}`;
    }
    questionChars.forEach((span, i) => {
      span.classList.toggle("is-visible", elapsed >= 400 + (i + 1) * 55);
    });
    const typed = Math.max(0, Math.floor((elapsed - timing.baselineStart) / 42));
    const progress = Math.max(0, Math.min(1, (elapsed - timing.steeringStart) / 2400));
    const front = progress * (responseChars.length + 5);
    responseChars.forEach((span, i) => {
      const changing = phase === "steering" && i <= front && i > front - 5;
      let char = baselineChars[i] || "";
      if (phase === "steered" || (phase === "steering" && i <= front - 5)) char = targetChars[i] || "";
      if (changing) char = glyphs[(i * 7 + Math.floor(elapsed / 65)) % glyphs.length];
      if (span.textContent !== char) span.textContent = char;
      span.classList.toggle("is-visible", phase === "steering" || phase === "steered" || i < typed);
      span.classList.toggle("is-glitch", changing);
    });
  }

  function stopFrame() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    previousTime = null;
  }

  function tick(now) {
    frame = null;
    if (paused || !inView || document.hidden || reducedMotion.matches) return;
    if (previousTime !== null) elapsed += Math.min(now - previousTime, 100);
    previousTime = now;
    if (elapsed > timing.end) elapsed = 0;
    render();
    frame = requestAnimationFrame(tick);
  }

  function resume() {
    if (!paused && inView && !document.hidden && !reducedMotion.matches && frame === null) {
      previousTime = null;
      frame = requestAnimationFrame(tick);
    }
  }

  function replay() {
    stopFrame();
    elapsed = reducedMotion.matches ? timing.steeringEnd : 0;
    paused = false;
    started = true;
    box.dataset.paused = "false";
    pauseButton.textContent = "Pause";
    pauseButton.setAttribute("aria-label", "Pause animation");
    render();
    resume();
  }

  pauseButton.addEventListener("click", () => {
    paused = !paused;
    box.dataset.paused = String(paused);
    pauseButton.textContent = paused ? "Play" : "Pause";
    pauseButton.setAttribute("aria-label", paused ? "Play animation" : "Pause animation");
    if (paused) stopFrame(); else resume();
  });
  box.querySelector(".steering-demo-replay").addEventListener("click", replay);
  new IntersectionObserver(([entry]) => {
    inView = entry.isIntersecting;
    if (inView && !started) replay();
    else if (inView) resume();
    else stopFrame();
  }, { threshold: .25 }).observe(box);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopFrame(); else resume();
  });
  reducedMotion.addEventListener("change", () => {
    pauseButton.hidden = reducedMotion.matches;
    replay();
  });
  // Scoped integration point for later benchmark text; no tab UI or page mutations.
  box.steeringDemo = {
    replay,
    setExample(next) {
      if (!["question", "baseline", "steered"].every((key) => typeof next[key] === "string" && next[key].trim())) {
        throw new TypeError("Each example needs question, baseline, and steered text.");
      }
      example = { question: next.question, baseline: next.baseline, steered: next.steered };
      prepare();
      replay();
    },
  };
  pauseButton.hidden = reducedMotion.matches;
  prepare();
})();
