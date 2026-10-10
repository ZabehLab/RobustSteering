"use strict";

(() => {
  const box = document.querySelector(".steering-demo");
  if (!box) return;
  // Schematic example only: these are not recorded benchmark/model outputs.
  // Replace this data with approved question/response pairs in the next iteration.
  let example = {
    benchmark: "TruthfulQA",
    question: "¿Los murciélagos son ciegos?",
    baseline: "Yes, bats are blind.",
    steered: "No. Bats can see; they also use echolocation to navigate.",
    metrics: ["Truthfulness", "Informativeness"],
  };
  const find = (name) => box.querySelector(".steering-demo-" + name);
  const question = find("question-text");
  const response = find("answer-text");
  const description = find("description");
  const pauseButton = find("pause");
  const scene = find("scene");
  const state = find("state");
  const direction = find("direction");
  const directionHead = find("direction-head");
  const directionLabels = find("direction-labels");
  const shiftMask = find("shift-mask");
  const dot = find("state-dot");
  const ripple = find("state-ripple");
  const travelBolt = find("travel-bolt");
  const metricNodes = [find("truth-label"), find("truth-arrow"), find("info-label"), find("info-arrow")];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const glyphs = Array.from("▓▒░<>/\\#¿!?");
  // All times refer to this one pausable timeline (milliseconds).
  const intro = {
    field: 400, direction: 1100, store: 2800, stored: 3900,
    shift: 4500, shifted: 5600, dot: 5800, expand: 6500, chat: 7550,
  };
  let questionChars, baselineChars, targetChars, responseChars, timing;
  let elapsed = 0;
  let previousTime = null;
  let frame = null;
  let inView = false;
  let paused = false;
  let started = false;
  let flightStart, flightEnd;
  const clamp = (x) => Math.max(0, Math.min(1, x));
  const progress = (start, duration) => clamp((elapsed - start) / duration);
  const ease = (x) => x * x * (3 - 2 * x);
  const reveal = (node, amount, distance = 0) => {
    node.style.opacity = amount;
    node.style.transform = "translateY(" + ((1 - amount) * distance) + "px)";
  };

  function characterNodes(node, count) {
    const nodes = Array.from({ length: count }, () => {
      const span = document.createElement("span");
      span.className = "steering-demo-char";
      return span;
    });
    node.replaceChildren(...nodes);
    return nodes;
  }

  function measureFlight() {
    const bounds = box.getBoundingClientRect();
    const from = find("stored-bolt").getBoundingClientRect();
    const to = find("bolt").getBoundingClientRect();
    flightStart = { x: from.x + from.width / 2 - bounds.x, y: from.y + from.height / 2 - bounds.y };
    flightEnd = { x: to.x + to.width / 2 - bounds.x, y: to.y + to.height / 2 - bounds.y };
  }

  function reserveResponseHeight() {
    // Measure both complete texts at the current width; keep the chat anchored.
    response.style.removeProperty("--demo-response-height");
    const sample = response.cloneNode(false);
    sample.style.cssText = "position:absolute;visibility:hidden;pointer-events:none;min-height:0;width:" + response.clientWidth + "px";
    response.parentElement.append(sample);
    const height = Math.max(...[example.baseline, example.steered].map((text) => {
      sample.textContent = text;
      return sample.getBoundingClientRect().height;
    }));
    response.style.setProperty("--demo-response-height", (height + parseFloat(getComputedStyle(response).lineHeight)) + "px");
    sample.remove();
    measureFlight();
  }

  function prepare() {
    const q = Array.from(example.question);
    baselineChars = Array.from(example.baseline);
    targetChars = Array.from(example.steered);
    questionChars = characterNodes(question, q.length);
    questionChars.forEach((span, i) => { span.textContent = q[i]; });
    responseChars = characterNodes(response, Math.max(baselineChars.length, targetChars.length));
    find("benchmark").textContent = example.benchmark;
    metricNodes[0].textContent = example.metrics[0];
    metricNodes[2].textContent = example.metrics[1];
    const questionStart = intro.chat + 250;
    const questionEnd = questionStart + q.length * 55;
    const baselineStart = questionEnd + 450;
    const baselineEnd = baselineStart + baselineChars.length * 42;
    const steeringStart = baselineEnd + 1000;
    const steeringEnd = steeringStart + 2800;
    timing = { questionStart, baselineStart, baselineEnd, steeringStart, steeringEnd, end: steeringEnd + 6800 };
    elapsed = 0;
    previousTime = null;
    render();
    reserveResponseHeight();
  }

  function phaseAt(time) {
    if (time < timing.questionStart) return "waiting";
    if (time < timing.baselineStart) return "question";
    if (time < timing.baselineEnd) return "baseline";
    if (time < timing.steeringStart) return "baseline-hold";
    if (time < timing.steeringEnd) return "steering";
    return "steered";
  }

  function stageAt(time) {
    if (time < intro.field) return "empty";
    if (time < intro.direction) return "english";
    if (time < intro.store) return "direction";
    if (time < intro.stored) return "store";
    if (time < intro.shift) return "stored";
    if (time < intro.shifted) return "shift";
    if (time < intro.expand) return "spanish";
    if (time < intro.chat) return "expand";
    return "conversation";
  }

  function renderState() {
    box.dataset.stage = stageAt(elapsed);
    state.style.opacity = progress(intro.field, 450);
    const drawn = progress(intro.direction, 850);
    const stored = ease(progress(intro.store, intro.stored - intro.store));
    direction.style.opacity = drawn * (1 - stored);
    directionHead.style.opacity = clamp(drawn * 4 - 3);
    directionLabels.style.opacity = drawn * (1 - progress(intro.store, 300));
    box.style.setProperty("--demo-direction", drawn);
    const scale = 1 - stored * .94;
    // Collapse around the vector midpoint, then carry it down into the controller.
    direction.setAttribute("transform", "translate(" + (169 * (1 - scale)) + " " + (160 * (1 - scale) + 210 * stored) + ") scale(" + scale + ")");
    const fill = progress(intro.store + 650, 450);
    box.style.setProperty("--demo-controller-fill", "rgba(0,124,124," + fill * .12 + ")");
    box.style.setProperty("--demo-controller-ink", fill > .5 ? "#007C7C" : "#919191");
    box.style.setProperty("--demo-controller-bolt", "rgb(" + Math.round(255 * (1 - fill)) + "," + Math.round(255 - 131 * fill) + "," + Math.round(255 - 131 * fill) + ")");
    const swipe = ease(progress(intro.shift, intro.shifted - intro.shift));
    box.style.setProperty("--demo-swipe", swipe);
    shiftMask.setAttribute("width", swipe * 280);
    const expand = ease(progress(intro.expand, intro.chat - intro.expand));
    const dotIn = progress(intro.dot, 250);
    dot.style.opacity = dotIn * (1 - expand);
    ripple.style.opacity = expand > 0 && expand < 1 ? 1 - expand : 0;
    ripple.setAttribute("r", 5 + expand * 180);
    scene.style.opacity = expand;
    scene.style.clipPath = expand < 1 ? "circle(" + expand * 150 + "% at 0% 50%)" : "none";
    // Transfer a visible copy of the stored controller to the model, not a new policy.
    if (flightStart && flightEnd && expand > 0 && expand < 1) {
      const x = flightStart.x + (flightEnd.x - flightStart.x) * expand;
      const y = flightStart.y + (flightEnd.y - flightStart.y) * expand - Math.sin(Math.PI * expand) * 70;
      travelBolt.style.transform = "translate(" + (x - 22) + "px," + (y - 28) + "px) scale(" + (.65 + .35 * expand) + ")";
      travelBolt.style.opacity = Math.min(1, expand * 8, (1 - expand) * 8);
    } else {
      travelBolt.style.opacity = 0;
    }
  }

  function render() {
    renderState();
    const phase = phaseAt(elapsed);
    if (box.dataset.phase !== phase) {
      box.dataset.phase = phase;
      if (phase === "baseline-hold") description.textContent = "Illustrative question: " + example.question + " Original response: " + example.baseline;
      if (phase === "steered") description.textContent = "Illustrative steered response: " + example.steered;
    }
    questionChars.forEach((span, i) => {
      span.classList.toggle("is-visible", elapsed >= timing.questionStart + (i + 1) * 55);
    });
    const typed = Math.max(0, Math.floor((elapsed - timing.baselineStart) / 42));
    const corruption = progress(timing.steeringStart, timing.steeringEnd - timing.steeringStart);
    const front = corruption * (responseChars.length + 5);
    responseChars.forEach((span, i) => {
      const changing = phase === "steering" && i <= front && i > front - 5;
      let char = baselineChars[i] || "";
      if (phase === "steered" || (phase === "steering" && i <= front - 5)) char = targetChars[i] || "";
      if (changing) char = glyphs[(i * 7 + Math.floor(elapsed / 65)) % glyphs.length];
      if (span.textContent !== char) span.textContent = char;
      span.classList.toggle("is-visible", phase === "steering" || phase === "steered" || i < typed);
      span.classList.toggle("is-glitch", changing);
    });
    // Each label appears before its arrow; these are qualitative schematic indicators.
    [350, 850, 1350, 1850].forEach((offset, i) => {
      reveal(metricNodes[i], ease(progress(timing.steeringEnd + offset, 300)), i % 2 ? 9 : 5);
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
    elapsed = reducedMotion.matches ? timing.steeringEnd + 2400 : 0;
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
  find("replay").addEventListener("click", replay);
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
  new ResizeObserver(reserveResponseHeight).observe(response.parentElement);
  // Reuse the same timeline/layout for approved benchmark examples, without tabs.
  box.steeringDemo = {
    replay,
    setExample(next) {
      if (!["question", "baseline", "steered"].every((key) => typeof next[key] === "string" && next[key].trim())) {
        throw new TypeError("Each example needs question, baseline, and steered text.");
      }
      if (next.metrics && (!Array.isArray(next.metrics) || next.metrics.length !== 2 || !next.metrics.every((s) => typeof s === "string" && s.trim()))) {
        throw new TypeError("Metrics must be two nonempty labels.");
      }
      example = { ...example, question: next.question, baseline: next.baseline, steered: next.steered, benchmark: next.benchmark || example.benchmark, metrics: next.metrics || example.metrics };
      prepare();
      replay();
    },
  };
  pauseButton.hidden = reducedMotion.matches;
  prepare();
})();
