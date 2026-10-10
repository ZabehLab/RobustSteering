"use strict";

(() => {
  const box = document.querySelector(".steering-demo");
  if (!box) return;
  // Schematic example only: these are not recorded benchmark/model outputs.
  // Replace this data with approved question/response pairs in the next iteration.
  let example = {
    benchmark: "An Example of Multilingual Grade-School Mathematics",
    question: "阿米娜有3盒铅笔，每盒有12支。她一共有多少支铅笔？",
    baseline: "阿米娜一共有36支铅笔。",
    steered: "Amina tiene 36 lápices en total.",
    metrics: ["Mathematical accuracy", "Spanish output"],
    symbols: ["=", "↑"],
  };
  const find = (name) => box.querySelector(".steering-demo-" + name);
  const question = find("question-text");
  const response = find("answer-text");
  const description = find("description");
  const pauseButton = find("pause");
  const scene = find("scene");
  const sceneFrame = find("scene-frame");
  const state = find("state");
  const direction = find("direction");
  const directionLine = find("direction-line");
  const directionHead = find("direction-head");
  const directionLabels = find("direction-labels");
  const shiftMask = find("shift-mask");
  const dot = find("state-dot");
  const ripple = find("state-ripple");
  const travelBolt = find("travel-bolt");
  const metricNodes = [find("truth-label"), find("truth-arrow"), find("info-label"), find("info-arrow")];
  const metricBoxes = [...box.querySelectorAll(".steering-demo-metric")];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const glyphs = Array.from("▓▒░<>/\\#¿!?");
  // All times refer to this one pausable timeline (milliseconds).
  const intro = {
    field: 400, endpoints: 1100, direction: 1550, store: 2800, stored: 3900,
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
  let sceneScale = 1;
  const clamp = (x) => Math.max(0, Math.min(1, x));
  const progress = (start, duration) => clamp((elapsed - start) / duration);
  const ease = (x) => x * x * (3 - 2 * x);
  const reveal = (node, amount, distance = 0) => {
    node.style.opacity = amount;
    node.style.transform = "translateY(" + ((1 - amount) * distance) + "px)";
  };

  function makeStateContours(node, seed, components, color) {
    // Seeded Gaussian-mixture samples and a Gaussian KDE, not hand-drawn ellipses.
    // Synthetic illustration only: these are not recorded model activations.
    const random = () => {
      seed |= 0;
      seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
    const totalWeight = components.reduce((sum, c) => sum + c.weight, 0);
    const samples = Array.from({ length: 600 }, () => {
      let draw = random() * totalWeight;
      const c = components.find((component) => (draw -= component.weight) < 0) || components[components.length - 1];
      const radius = Math.sqrt(-2 * Math.log(Math.max(random(), 1e-10)));
      const theta = 2 * Math.PI * random();
      const a = radius * Math.cos(theta);
      const b = radius * Math.sin(theta);
      return [c.x + c.sx * a, c.y + c.sy * (c.rho * a + Math.sqrt(1 - c.rho * c.rho) * b)];
    });
    // Evaluate beyond the visible square, so contours can cross its edges cleanly.
    const n = 128;
    const density = new Float64Array(n * n);
    const bandwidth = .45;
    let peak = 0;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const px = -6 + x * 12 / (n - 1);
        const py = 6 - y * 12 / (n - 1);
        let value = 0;
        for (const [sx, sy] of samples) {
          const distance = ((px - sx) ** 2 + (py - sy) ** 2) / (2 * bandwidth * bandwidth);
          if (distance < 18) value += Math.exp(-distance);
        }
        density[y * n + x] = value;
        peak = Math.max(peak, value);
      }
    }
    const cases = [[], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], [[3, 0], [1, 2]], [[0, 2]], [[3, 2]], [[2, 3]], [[0, 2]], [[0, 1], [2, 3]], [[1, 2]], [[1, 3]], [[0, 1]], [[3, 0]], []];
    const paths = [];
    for (const fraction of [.04, .08, .14, .23, .34, .47, .61, .76, .9]) {
      const threshold = peak * fraction;
      const vertices = new Map();
      const adjacency = new Map();
      const segments = [];
      for (let y = 0; y < n - 1; y++) {
        for (let x = 0; x < n - 1; x++) {
          const values = [density[y * n + x], density[y * n + x + 1], density[(y + 1) * n + x + 1], density[(y + 1) * n + x]];
          const mask = values.reduce((code, v, i) => code | (v >= threshold ? 1 << i : 0), 0);
          if (!cases[mask].length) continue;
          const corners = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]];
          const keys = ["h:" + x + ":" + y, "v:" + (x + 1) + ":" + y, "h:" + x + ":" + (y + 1), "v:" + x + ":" + y];
          const vertex = (edge) => {
            const key = keys[edge];
            if (!vertices.has(key)) {
              const next = (edge + 1) % 4;
              const t = (threshold - values[edge]) / (values[next] - values[edge]);
              const [ax, ay] = corners[edge];
              const [bx, by] = corners[next];
              vertices.set(key, [-70 + (ax + (bx - ax) * t) * 480 / (n - 1), -82 + (ay + (by - ay) * t) * 480 / (n - 1)]);
            }
            return key;
          };
          for (const [a, b] of cases[mask]) {
            const pair = [vertex(a), vertex(b)];
            const index = segments.push(pair) - 1;
            for (const key of pair) {
              if (!adjacency.has(key)) adjacency.set(key, []);
              adjacency.get(key).push(index);
            }
          }
        }
      }
      const used = new Set();
      let d = "";
      segments.forEach(([start, end], index) => {
        if (used.has(index)) return;
        const points = [vertices.get(start), vertices.get(end)];
        used.add(index);
        let key = end;
        while (key !== start) {
          const nextIndex = adjacency.get(key).find((i) => !used.has(i));
          if (nextIndex === undefined) break;
          used.add(nextIndex);
          const pair = segments[nextIndex];
          key = pair[0] === key ? pair[1] : pair[0];
          points.push(vertices.get(key));
        }
        // Closed rings get a very light density fill; open boundary contours stay unfilled.
        if (key !== start) return;
        d += "M" + points.map(([x, y]) => x.toFixed(2) + "," + y.toFixed(2)).join("L") + "Z";
      });
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", d);
      path.setAttribute("fill", color);
      path.setAttribute("fill-opacity", ".055");
      path.setAttribute("fill-rule", "evenodd");
      path.setAttribute("stroke", color);
      path.setAttribute("stroke-opacity", ".6");
      path.setAttribute("stroke-width", "1.1");
      path.setAttribute("stroke-linejoin", "round");
      paths.push(path);
    }
    node.replaceChildren(...paths);
    node.dataset.sampleCount = samples.length;
    node.dataset.method = "Seeded Gaussian-mixture sampling; Gaussian KDE; marching-squares contours";
  }

  // Compute once, then reuse exactly the same fields for every replay.
  // The multi-modal landscapes are independent of the two language-direction points.
  makeStateContours(find("field-english"), 2027, [
    { weight: .25, x: 0, y: 0, sx: 2.6, sy: 2.6, rho: -.1 },
    { weight: .13, x: -2.4, y: 1.5, sx: .75, sy: .9, rho: -.3 },
    { weight: .1, x: -.45, y: 2.2, sx: 1.0, sy: .55, rho: -.2 },
    { weight: .13, x: 2.0, y: 1.5, sx: .8, sy: .8, rho: .2 },
    { weight: .12, x: -1.75, y: -.7, sx: .9, sy: .65, rho: .1 },
    { weight: .1, x: .7, y: .1, sx: .75, sy: .9, rho: -.4 },
    { weight: .08, x: 2.5, y: -1.5, sx: .7, sy: .8, rho: -.2 },
    { weight: .09, x: -.4, y: -2.4, sx: 1.15, sy: .55, rho: .1 },
  ], "#919191");
  makeStateContours(find("field-chinese"), 2028, [
    { weight: .3, x: 0, y: 0, sx: 2.9, sy: 2.9, rho: .08 },
    { weight: .13, x: -1.7, y: 2.1, sx: .85, sy: 1.0, rho: .5 },
    { weight: .16, x: 1.2, y: -1.6, sx: 1.1, sy: .8, rho: .5 },
    { weight: .1, x: .7, y: 1.3, sx: .8, sy: .55, rho: -.45 },
    { weight: .07, x: -1.6, y: -1.7, sx: .9, sy: .5, rho: -.2 },
    { weight: .06, x: 2.8, y: 1.8, sx: .55, sy: .9, rho: -.3 },
    { weight: .05, x: -3.1, y: -.1, sx: .7, sy: .85, rho: .2 },
    { weight: .04, x: -.2, y: -.2, sx: .4, sy: .4, rho: -.1 },
    { weight: .035, x: -2.9, y: 2.5, sx: .4, sy: .4, rho: .2 },
    { weight: .03, x: 2.1, y: -3, sx: .7, sy: .45, rho: .15 },
    { weight: .025, x: 2.7, y: -.1, sx: .45, sy: .6, rho: -.2 },
  ], "#8B1E1E");

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
    box.style.setProperty("--demo-svg-scale-inverse", 340 / find("state-map").clientWidth);
    sceneScale = parseFloat(getComputedStyle(box).getPropertyValue("--demo-chat-scale"));
    const model = find("robot").getBoundingClientRect();
    find("stimulation").style.transform = "translate(" + (model.x + model.width / 2 - bounds.x - 45 * sceneScale) + "px," + (model.y - bounds.y - 72 * sceneScale) + "px) scale(" + sceneScale + ")";
    const from = find("stored-bolt").getBoundingClientRect();
    const to = find("bolt").getBoundingClientRect();
    flightStart = { x: from.x + from.width / 2 - bounds.x, y: from.y + from.height / 2 - bounds.y };
    flightEnd = { x: to.x + to.width / 2 - bounds.x, y: to.y + to.height / 2 - bounds.y };
  }

  function reserveResponseHeight() {
    // Reserve every mixed/glitched response too, not only the two complete answers.
    // Inline-block glitch glyphs can add a line; they must not resize the scene.
    const sample = response.cloneNode(false);
    sample.style.cssText = "position:absolute;visibility:hidden;pointer-events:none;min-height:0;height:auto;width:" + response.clientWidth + "px";
    response.parentElement.append(sample);
    let height = Math.max(...[example.baseline, example.steered].map((text) => {
      sample.textContent = text;
      return sample.offsetHeight;
    }));
    const probe = document.createElement("span");
    probe.style.display = "inline-block";
    sample.replaceChildren(probe);
    let widestGlyph = glyphs[0], widestWidth = 0;
    glyphs.forEach((glyph) => {
      probe.textContent = glyph;
      if (probe.offsetWidth > widestWidth) {
        widestWidth = probe.offsetWidth;
        widestGlyph = glyph;
      }
    });
    const mixed = characterNodes(sample, responseChars.length);
    for (let front = 0; front <= mixed.length + 5; front++) {
      mixed.forEach((span, i) => {
        const changing = i <= front && i > front - 5;
        span.classList.add("is-visible");
        span.classList.toggle("is-glitch", changing);
        span.textContent = changing ? widestGlyph : i <= front - 5 ? targetChars[i] || "" : baselineChars[i] || "";
      });
      height = Math.max(height, sample.offsetHeight);
    }
    response.style.setProperty("--demo-response-height", height + "px");
    sample.remove();
    const scale = parseFloat(getComputedStyle(box).getPropertyValue("--demo-chat-scale"));
    sceneFrame.style.height = scene.offsetHeight * scale + "px";
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
    metricNodes[1].textContent = example.symbols[0];
    metricNodes[3].textContent = example.symbols[1];
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
    if (time < intro.endpoints) return "english";
    if (time < intro.direction) return "endpoints";
    if (time < intro.store) return "direction";
    if (time < intro.stored) return "store";
    if (time < intro.shift) return "stored";
    if (time < intro.shifted) return "shift";
    if (time < intro.expand) return "chinese";
    if (time < intro.chat) return "expand";
    return "conversation";
  }

  function renderState() {
    box.dataset.stage = stageAt(elapsed);
    state.style.opacity = progress(intro.field, 450);
    const points = progress(intro.endpoints, 300);
    const drawn = progress(intro.direction, 850);
    const stored = ease(progress(intro.store, intro.stored - intro.store));
    direction.style.opacity = points * (1 - stored);
    directionLine.style.opacity = drawn;
    directionHead.style.opacity = clamp(drawn * 4 - 3);
    directionLabels.style.opacity = points * (1 - progress(intro.store, 300));
    box.style.setProperty("--demo-direction", drawn);
    const scale = 1 - stored * .94;
    // Collapse around the vector midpoint, then carry it down into the controller.
    direction.setAttribute("transform", "translate(" + (169 * (1 - scale) - 61 * stored) + " " + (160 * (1 - scale) + 210 * stored) + ") scale(" + scale + ")");
    const fill = progress(intro.store + 650, 450);
    box.style.setProperty("--demo-controller-fill", "rgba(0,124,124," + fill * .12 + ")");
    box.style.setProperty("--demo-controller-ink", fill > .5 ? "#007C7C" : "#919191");
    box.style.setProperty("--demo-controller-bolt", "rgb(" + Math.round(255 * (1 - fill)) + "," + Math.round(255 - 131 * fill) + "," + Math.round(255 - 131 * fill) + ")");
    const swipe = ease(progress(intro.shift, intro.shifted - intro.shift));
    box.style.setProperty("--demo-swipe", swipe);
    shiftMask.setAttribute("width", swipe * 280);
    const expand = ease(progress(intro.expand, intro.chat - intro.expand));
    find("stimulation").style.opacity = elapsed >= intro.chat ? 1 : 0;
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
      travelBolt.style.transform = "translate(" + (x - 22) + "px," + (y - 28) + "px) scale(" + (.65 + (sceneScale - .65) * expand) + ")";
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
    metricBoxes.forEach((node, i) => {
      node.style.opacity = ease(progress(timing.steeringEnd + 350 + i * 1000, 300));
    });
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
  new ResizeObserver(reserveResponseHeight).observe(sceneFrame);
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
      if (next.symbols && (!Array.isArray(next.symbols) || next.symbols.length !== 2 || !next.symbols.every((s) => typeof s === "string"))) {
        throw new TypeError("Symbols must be two strings.");
      }
      example = { ...example, question: next.question, baseline: next.baseline, steered: next.steered, benchmark: next.benchmark || example.benchmark, metrics: next.metrics || example.metrics, symbols: next.symbols || example.symbols };
      prepare();
      replay();
    },
  };
  pauseButton.hidden = reducedMotion.matches;
  prepare();
})();
