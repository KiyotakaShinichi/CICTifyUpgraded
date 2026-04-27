const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const micBtn = document.getElementById("mic-btn");
const chatbox = document.getElementById("chatbox");
const expandBtn = document.getElementById("expand-btn");
const routeFeatureBtn = document.getElementById("route-feature-btn");
const routePanel = document.getElementById("route-panel");
const routeBtn = document.getElementById("route-btn");
const routeFrom = document.getElementById("route-from");
const routeTo = document.getElementById("route-to");
const routeAlgorithm = document.getElementById("route-algorithm");
const routeSummary = document.getElementById("route-summary");
const routeMapToggle = document.getElementById("route-map-toggle");
const routeCanvasWrap = document.getElementById("route-canvas-wrap");
const routePlanBg = document.getElementById("route-plan-bg");
const routeSvg = document.getElementById("route-svg");

const ROUTE_VIEW_WIDTH = 1000;
const ROUTE_VIEW_HEIGHT = 700;

let abortController = null;
let recognizing = false;
let recognition;
let routeOptionsCache = [];
let hasDrawableRoute = false;

function escapeHtml(text) {
  return String(text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatInlineMarkdown(text) {
  let html = String(text || "");
  html = html.replace(/\*\*\*([^*]+)\*\*\*/g, "<strong><em>$1</em></strong>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  html = html.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  html = html.replace(/_([^_]+)_/g, "<em>$1</em>");
  return html;
}

function renderRichText(raw) {
  const safe = escapeHtml(raw).replace(/\r\n?/g, "\n");
  const lines = safe.split("\n");

  const out = [];
  let listItems = [];
  const flushList = () => {
    if (!listItems.length) return;
    out.push(`<ul>${listItems.map((item) => `<li>${formatInlineMarkdown(item)}</li>`).join("")}</ul>`);
    listItems = [];
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      out.push("<br>");
      continue;
    }

    const bulletMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (bulletMatch) {
      listItems.push(bulletMatch[1]);
      continue;
    }

    flushList();
    out.push(formatInlineMarkdown(trimmed));
  }

  flushList();
  const html = out.join("<br>").replace(/(?:<br>){3,}/g, "<br><br>");
  return html;
}

document.addEventListener("DOMContentLoaded", () => {
  const chatLogEl = document.getElementById("chat-log");
  if (!chatLogEl) return;



  const hours = new Date().getHours();
  const timeGreeting =
    hours < 12 ? "Good morning" :
    hours < 18 ? "Good afternoon" :
    "Good evening";

  const botMsg = document.createElement("div");
  botMsg.classList.add("chat-message", "bot-message");
  botMsg.innerHTML = `
    <img src="images/CICTify_ChatLogo.png" alt="Bot Avatar">
    <div class="text">👋 ${timeGreeting}! I’m your CICT Chatbot. How can I help you today?</div>
  `;

  setTimeout(() => {
    chatLogEl.appendChild(botMsg);
    chatLogEl.scrollTop = chatLogEl.scrollHeight;
  }, 150);

  loadRouteOptions();
});

// Toggle chatbox visibility
function toggleChat() {
  chatbox.classList.toggle("show");
}

// Typing indicator
function showTypingIndicator() {
  const typingIndicator = document.createElement("div");
  typingIndicator.classList.add("typing-indicator");
  typingIndicator.innerHTML = `
    <img src="images/CICTify_ChatLogo.png" alt="Bot Avatar">
    <div class="typing-dots">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  const chatLog = document.getElementById("chat-log");
  chatLog.appendChild(typingIndicator);
  chatLog.scrollTop = chatLog.scrollHeight;
  return typingIndicator;
}

function hideTypingIndicator(typingIndicator) {
  if (typingIndicator && typingIndicator.parentNode) typingIndicator.remove();
}

function appendBotMessage(text, enableTts = true) {
  const chatLog = document.getElementById("chat-log");
  if (!chatLog) return;

  const botMsg = document.createElement("div");
  botMsg.classList.add("chat-message", "bot-message");
  botMsg.innerHTML = `
    <img src="images/CICTify_ChatLogo.png" alt="Bot Avatar">
    <div class="text"></div>
  `;
  botMsg.querySelector(".text").innerHTML = renderRichText(text || "");

  if (enableTts) {
    const listenBtn = document.createElement("button");
    listenBtn.textContent = "🔊 Listen";
    listenBtn.className = "tts-btn";
    listenBtn.style.marginTop = "5px";
    listenBtn.onclick = () => {
      const utterance = new SpeechSynthesisUtterance(String(text || ""));
      utterance.rate = 1;
      utterance.pitch = 1;
      utterance.volume = 1;
      window.speechSynthesis.speak(utterance);
    };
    botMsg.appendChild(listenBtn);
  }

  chatLog.appendChild(botMsg);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setMapVisibility(visible) {
  if (!routeCanvasWrap || !routeMapToggle) return;
  routeCanvasWrap.classList.toggle("hidden", !visible);
  routeMapToggle.textContent = visible ? "Hide map" : "Show map";
}

function inferPlanBackground(route) {
  const explicit = String(route?.plan_background_url || "").trim();
  if (explicit) return explicit;
  const points = Array.isArray(route?.points) ? route.points : [];
  const buildings = points
    .map((p) => String(p?.building || "").toLowerCase())
    .filter(Boolean);

  const floors = points.map((p) => String(p?.floor || "").toLowerCase()).filter(Boolean);
  const hasCampus = buildings.some((b) => b.includes("main campus") || b.includes("bulsu"))
    || floors.some((f) => f.includes("campus"));
  const hasCict = buildings.some((b) => b.includes("pimentel") || b.includes("nstp") || b.includes("cict"));

  if (hasCampus) return "/plan_assets/bulsu-main-campus-campus.png";
  // Indoor building routes should not be forced to the campus image.
  if (hasCict) return "";
  return "";
}

function setPlanBackground(route) {
  if (!routePlanBg) return;
  const src = inferPlanBackground(route);
  if (!src) {
    routePlanBg.classList.add("hidden");
    routePlanBg.removeAttribute("src");
    return;
  }
  routePlanBg.src = src;
  routePlanBg.classList.remove("hidden");
}

function drawSvgBackground(route) {
  if (!routeSvg) return;
  const src = inferPlanBackground(route);
  if (!src) return;
  routeSvg.setAttribute("viewBox", `0 0 ${ROUTE_VIEW_WIDTH} ${ROUTE_VIEW_HEIGHT}`);
  routeSvg.setAttribute("preserveAspectRatio", "none");

  const bg = document.createElementNS("http://www.w3.org/2000/svg", "image");
  bg.setAttribute("href", src);
  bg.setAttributeNS("http://www.w3.org/1999/xlink", "xlink:href", src);
  bg.setAttribute("x", "0");
  bg.setAttribute("y", "0");
  bg.setAttribute("width", String(ROUTE_VIEW_WIDTH));
  bg.setAttribute("height", String(ROUTE_VIEW_HEIGHT));
  bg.setAttribute("preserveAspectRatio", "none");
  bg.setAttribute("opacity", "0.92");
  bg.style.imageRendering = "auto";
  routeSvg.appendChild(bg);
}

function clearPlanBackground() {
  if (!routePlanBg) return;
  routePlanBg.classList.add("hidden");
  routePlanBg.removeAttribute("src");
}

function drawMapPlaceholder(message) {
  if (!routeSvg) return;
  clearPlanBackground();
  routeSvg.innerHTML = "";
  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", "50%");
  text.setAttribute("y", "50%");
  text.setAttribute("text-anchor", "middle");
  text.setAttribute("dominant-baseline", "middle");
  text.setAttribute("font-size", "14");
  text.setAttribute("fill", "#6b7280");
  text.textContent = message || "No map preview available.";
  routeSvg.appendChild(text);
}

function drawSchematicRoute(points, route = null) {
  if (!routeSvg) return;
  routeSvg.setAttribute("viewBox", `0 0 ${ROUTE_VIEW_WIDTH} ${ROUTE_VIEW_HEIGHT}`);
  routeSvg.setAttribute("preserveAspectRatio", "none");
  routeSvg.innerHTML = "";
  if (route) {
    drawSvgBackground(route);
  }

  const width = ROUTE_VIEW_WIDTH;
  const height = ROUTE_VIEW_HEIGHT;
  const padding = 70;
  const count = Math.max(1, points.length);
  const stepX = count > 1 ? (width - padding * 2) / (count - 1) : 0;

  const mapped = points.map((p, idx) => ({
    ...p,
    sx: padding + stepX * idx,
    sy: height / 2 + ((idx % 2 === 0) ? -40 : 40),
  }));

  const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  poly.setAttribute("fill", "none");
  poly.setAttribute("stroke", "#f6921e");
  poly.setAttribute("stroke-width", "6");
  poly.setAttribute("stroke-linecap", "round");
  poly.setAttribute("stroke-linejoin", "round");
  poly.setAttribute("points", mapped.map((p) => `${p.sx},${p.sy}`).join(" "));
  routeSvg.appendChild(poly);

  animateRoutePolyline(poly);
  animateMovingDot(poly);

  mapped.forEach((p, idx) => {
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", String(p.sx));
    dot.setAttribute("cy", String(p.sy));
    dot.setAttribute("r", idx === 0 || idx === mapped.length - 1 ? "8" : "5");
    dot.setAttribute("fill", idx === 0 ? "#2a9d8f" : idx === mapped.length - 1 ? "#e63946" : "#264653");
    routeSvg.appendChild(dot);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", String(p.sx + 10));
    label.setAttribute("y", String(p.sy - 10));
    label.setAttribute("font-size", "13");
    label.setAttribute("fill", "#1f2937");
    label.textContent = String(p.name || p.id || "Node");
    routeSvg.appendChild(label);
  });
}

function animateRoutePolyline(polyline) {
  if (!polyline || typeof polyline.getTotalLength !== "function") return;
  const total = polyline.getTotalLength();
  if (!Number.isFinite(total) || total <= 0) return;

  polyline.style.strokeDasharray = `${total}`;
  polyline.style.strokeDashoffset = `${total}`;
  polyline.style.transition = "stroke-dashoffset 1.1s ease-out";

  requestAnimationFrame(() => {
    polyline.style.strokeDashoffset = "0";
  });
}

function animateMovingDot(polyline) {
  if (!routeSvg || !polyline || typeof polyline.getTotalLength !== "function") return;
  const total = polyline.getTotalLength();
  if (!Number.isFinite(total) || total <= 0) return;

  const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  dot.setAttribute("r", "6");
  dot.setAttribute("fill", "#2563eb");
  dot.setAttribute("stroke", "#ffffff");
  dot.setAttribute("stroke-width", "2");
  routeSvg.appendChild(dot);

  const durationMs = 1800;
  let startedAt = null;

  const tick = (ts) => {
    if (startedAt === null) startedAt = ts;
    const elapsed = ts - startedAt;
    const progress = Math.min(1, elapsed / durationMs);
    const point = polyline.getPointAtLength(total * progress);
    dot.setAttribute("cx", String(point.x));
    dot.setAttribute("cy", String(point.y));
    if (progress < 1) {
      requestAnimationFrame(tick);
    }
  };

  requestAnimationFrame(tick);
}

function drawRoute(route) {
  if (!routeSvg) return;
  routeSvg.setAttribute("viewBox", `0 0 ${ROUTE_VIEW_WIDTH} ${ROUTE_VIEW_HEIGHT}`);
  routeSvg.setAttribute("preserveAspectRatio", "none");
  routeSvg.innerHTML = "";
  clearPlanBackground();
  drawSvgBackground(route);

  const points = Array.isArray(route?.points) ? route.points : [];
  if (!points.length) {
    hasDrawableRoute = false;
    setMapVisibility(false);
    if (routeMapToggle) routeMapToggle.disabled = false;
    routeSummary.textContent = "No drawable route points available.";
    clearPlanBackground();
    return;
  }

  const usable = points.filter((p) => Number.isFinite(p?.x) && Number.isFinite(p?.y));
  if (!usable.length || usable.length !== points.length) {
    hasDrawableRoute = true;
    setMapVisibility(true);
    if (routeMapToggle) routeMapToggle.disabled = false;
    drawSchematicRoute(points, route);
    const pathLabel = Array.isArray(route.path_names) ? route.path_names.join(" -> ") : "";
    routeSummary.textContent = `Route ready. Showing live schematic map for the full path.${pathLabel ? ` Path: ${pathLabel}` : ""}`;
    return;
  }

  hasDrawableRoute = true;
  if (routeMapToggle) routeMapToggle.disabled = false;
  setMapVisibility(true);

  const xs = usable.map((p) => Number(p.x));
  const ys = usable.map((p) => Number(p.y));
  const bounds = route?.coordinate_bounds || {};
  const minX = Number.isFinite(Number(bounds.min_x)) ? Number(bounds.min_x) : Math.min(...xs);
  const maxX = Number.isFinite(Number(bounds.max_x)) ? Number(bounds.max_x) : Math.max(...xs);
  const minY = Number.isFinite(Number(bounds.min_y)) ? Number(bounds.min_y) : Math.min(...ys);
  const maxY = Number.isFinite(Number(bounds.max_y)) ? Number(bounds.max_y) : Math.max(...ys);

  const padding = 40;
  const width = ROUTE_VIEW_WIDTH;
  const height = ROUTE_VIEW_HEIGHT;
  const spanX = Math.max(1, maxX - minX);
  const spanY = Math.max(1, maxY - minY);

  const mapped = usable.map((p) => {
    const nx = ((Number(p.x) - minX) / spanX) * (width - padding * 2) + padding;
    const ny = ((Number(p.y) - minY) / spanY) * (height - padding * 2) + padding;
    return { ...p, sx: nx, sy: ny };
  });

  const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  poly.setAttribute("fill", "none");
  poly.setAttribute("stroke", "#f6921e");
  poly.setAttribute("stroke-width", "6");
  poly.setAttribute("stroke-linecap", "round");
  poly.setAttribute("stroke-linejoin", "round");
  poly.setAttribute("points", mapped.map((p) => `${p.sx},${p.sy}`).join(" "));
  routeSvg.appendChild(poly);

  animateRoutePolyline(poly);
  animateMovingDot(poly);

  mapped.forEach((p, idx) => {
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", String(p.sx));
    dot.setAttribute("cy", String(p.sy));
    dot.setAttribute("r", idx === 0 || idx === mapped.length - 1 ? "8" : "5");
    dot.setAttribute("fill", idx === 0 ? "#2a9d8f" : idx === mapped.length - 1 ? "#e63946" : "#264653");
    routeSvg.appendChild(dot);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", String(p.sx + 10));
    label.setAttribute("y", String(p.sy - 10));
    label.setAttribute("font-size", "13");
    label.setAttribute("fill", "#1f2937");
    label.textContent = String(p.name || p.id || "Node");
    routeSvg.appendChild(label);
  });

  const pathLabel = Array.isArray(route.path_names) ? route.path_names.join(" -> ") : "";
  routeSummary.textContent = `Algorithm: ${route.algorithm || "astar"} | Cost: ${route.distance ?? "n/a"}${pathLabel ? ` | Path: ${pathLabel}` : ""}`;
}

async function computeRoute() {
  const fromId = String(routeFrom?.value || "").trim();
  const toId = String(routeTo?.value || "").trim();
  const fromLabel = String(routeFrom?.selectedOptions?.[0]?.textContent || "").trim();
  const toLabel = String(routeTo?.selectedOptions?.[0]?.textContent || "").trim();
  const algorithm = String(routeAlgorithm?.value || "astar").trim().toLowerCase();
  if (!fromId || !toId) {
    routeSummary.textContent = "Please fill in both From and To.";
    return;
  }

  routeSummary.textContent = "Computing route...";
  try {
    const response = await fetch("/api/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from_id: fromId, to_id: toId, from_label: fromLabel, to_label: toLabel, algorithm }),
    });
    const data = await response.json();
    if (!response.ok || data.status !== "success") {
      hasDrawableRoute = false;
      if (routeMapToggle) routeMapToggle.disabled = false;
      drawMapPlaceholder("No map preview available.");
      setMapVisibility(false);
      routeSummary.textContent = data.message || "Route request failed.";
      appendBotMessage(`I couldn't compute a route yet. ${data.message || "Route request failed."}`, false);
      return;
    }
    const route = data.route || {};
    if (data.plan_background_url) {
      route.plan_background_url = String(data.plan_background_url);
    }
    drawRoute(route);
    const botReply = String(data.reply || "").trim() || (route.directions_text ? `🧭 Route guidance:\n${route.directions_text}` : "Route is ready.");
    appendBotMessage(botReply, true);
    routeSummary.textContent = `Route ready. Follow the guidance in chat. Algorithm: ${route.algorithm || "astar"}.`;
  } catch (error) {
    console.error("Route error:", error);
    hasDrawableRoute = false;
    if (routeMapToggle) routeMapToggle.disabled = false;
    drawMapPlaceholder("Unable to fetch map preview.");
    setMapVisibility(false);
    routeSummary.textContent = "Failed to connect to route service.";
    appendBotMessage("I couldn't reach the route service right now. Please try again.", false);
  }
}

async function loadRouteOptions() {
  if (!routeFrom || !routeTo) return;

  try {
    const response = await fetch("/api/route/options");
    const data = await response.json();
    if (!response.ok || data.status !== "success") {
      routeSummary.textContent = "Could not load route choices.";
      return;
    }

    const options = Array.isArray(data.options) ? data.options : [];
    routeOptionsCache = options;

    const renderGroupedOptions = (items) => {
      const groups = {
        cict_indoor: [],
        campus_place: [],
      };
      items.forEach((opt) => {
        const t = String(opt.place_type || "");
        if (groups[t]) {
          groups[t].push(opt);
        }
      });

      const makeOptions = (arr) => arr
        .map((opt) => {
          const id = String(opt.id || "");
          const label = String(opt.label || opt.name || id);
          return `<option value="${escapeHtml(id)}">${escapeHtml(label)}</option>`;
        })
        .join("");

      let html = "";
      if (groups.cict_indoor.length) {
        html += `<optgroup label="CICT Indoor (Rooms/Labs/Stairs)">${makeOptions(groups.cict_indoor)}</optgroup>`;
      }
      if (groups.campus_place.length) {
        html += `<optgroup label="Campus Places (Buildings/Gates/Areas)">${makeOptions(groups.campus_place)}</optgroup>`;
      }
      return html;
    };

    const grouped = renderGroupedOptions(options);
    routeFrom.innerHTML = `<option value="">From...</option>${grouped}`;
    routeTo.innerHTML = `<option value="">To...</option>${grouped}`;
    routeSummary.textContent = `Loaded ${options.length} route points. Select From and To.`;
  } catch (error) {
    console.error("Route options error:", error);
    routeSummary.textContent = "Failed to load route choices.";
  }
}

function selectedOptionById(nodeId) {
  return routeOptionsCache.find((opt) => String(opt.id) === String(nodeId));
}

function filterDestinationOptions() {
  if (!routeFrom || !routeTo) return;
  const fromId = String(routeFrom.value || "").trim();
  const from = selectedOptionById(fromId);

  const candidates = !from
    ? routeOptionsCache
    : routeOptionsCache.filter(
        (opt) =>
          opt.component === from.component &&
          String(opt.id) !== fromId &&
          String(opt.place_type || "") === String(from.place_type || "")
      );

  const currentTo = String(routeTo.value || "").trim();
  const groups = {
    cict_indoor: [],
    campus_place: [],
  };
  candidates.forEach((opt) => {
    const t = String(opt.place_type || "");
    if (groups[t]) groups[t].push(opt);
  });

  const makeOptions = (arr) => arr
    .map((opt) => {
      const id = String(opt.id || "");
      const label = String(opt.label || opt.name || id);
      return `<option value="${escapeHtml(id)}">${escapeHtml(label)}</option>`;
    })
    .join("");

  let groupedHtml = "";
  if (groups.cict_indoor.length) {
    groupedHtml += `<optgroup label="CICT Indoor (Rooms/Labs/Stairs)">${makeOptions(groups.cict_indoor)}</optgroup>`;
  }
  if (groups.campus_place.length) {
    groupedHtml += `<optgroup label="Campus Places (Buildings/Gates/Areas)">${makeOptions(groups.campus_place)}</optgroup>`;
  }

  routeTo.innerHTML = `<option value="">To...</option>${groupedHtml}`;
  if (currentTo && candidates.some((opt) => String(opt.id) === currentTo)) {
    routeTo.value = currentTo;
  }

  if (from) {
    const domainLabel = String(from.place_type || "") === "cict_indoor" ? "CICT-only" : "Campus-only";
    routeSummary.textContent = `Showing ${candidates.length} ${domainLabel} destinations from ${from.name}.`;
  }
}

// ---- Main Send Message ----
async function sendMessage() {
  const chatLog = document.getElementById("chat-log");
  const message = userInput.value.trim();
  if (message === "") return;

  // Disable input while processing
  userInput.disabled = true;
  sendBtn.disabled = true;
  micBtn.disabled = true;

  sendBtn.innerHTML = "⏹️";
  sendBtn.style.fontSize = "20px";
  sendBtn.style.background = "transparent";
  sendBtn.style.border = "none";

  const userMsg = document.createElement("div");
  userMsg.classList.add("chat-message", "user-message");
  userMsg.innerHTML = `
    <img src="images/CICTify_ChatLogo.png" alt="User Avatar">
    <div class="text"></div>
  `;
  userMsg.querySelector(".text").textContent = message;
  chatLog.appendChild(userMsg);
  chatLog.scrollTop = chatLog.scrollHeight;

  userInput.value = "";

  const typingIndicator = showTypingIndicator();

  abortController = new AbortController();
  const { signal } = abortController;

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal,
    });

    const data = await response.json();
    hideTypingIndicator(typingIndicator);

    const botReply = data.reply || "⚠️ No response received.";
    appendBotMessage(botReply, true);
  } catch (error) {
    hideTypingIndicator(typingIndicator);
    const botMsg = document.createElement("div");
    botMsg.classList.add("chat-message", "bot-message");

    if (error.name === "AbortError") {
      botMsg.innerHTML = `
        <img src="images/CICTify_ChatLogo.png" alt="Bot Avatar">
        <div class="text">🛑 Query stopped by user.</div>
      `;
    } else {
      console.error("Error sending message:", error);
      botMsg.innerHTML = `
        <img src="images/CICTify_ChatLogo.png" alt="Bot Avatar">
        <div class="text">⚠️ Server not responding. Please check your Flask app.</div>
      `;
    }
    chatLog.appendChild(botMsg);
    chatLog.scrollTop = chatLog.scrollHeight;
  } finally {
  userInput.disabled = false;
  sendBtn.disabled = false;
  micBtn.disabled = false;
  sendBtn.innerHTML = `<img src="images/sendIcon.png" alt="Send">`;
  abortController = null;
}

}

// ---- Stop Button ----
sendBtn.addEventListener("click", () => {
  if (abortController) {
    abortController.abort();
    return;
  }
  sendMessage();
  sendBtn.classList.add("send-clicked");
  setTimeout(() => sendBtn.classList.remove("send-clicked"), 250);
});

// ---- Voice Input ----
const correctionMap = {
  "bull sue": "BULSU",
  "bull zoo": "BULSU",
  "bulsu": "BULSU",
  "bull shoe": "BULSU",
  "bulls you": "BULSU",
  "ci ct": "CICT",
  "see ict": "CICT",
  "ci ctify": "CICTify",
  "bulsu cict": "BULSU CICT",
  "bulsu site": "BULSU CICT"
};

function correctTranscript(text) {
  let corrected = text.toLowerCase();
  for (const [wrong, right] of Object.entries(correctionMap)) {
    const pattern = new RegExp(`\\b${wrong}\\b`, "gi");
    corrected = corrected.replace(pattern, right);
  }
  return corrected.charAt(0).toUpperCase() + corrected.slice(1);
}

if ("webkitSpeechRecognition" in window) {
  recognition = new webkitSpeechRecognition();
  recognition.continuous = false;
  recognition.lang = "en-US";
  recognition.interimResults = false;

  recognition.onstart = () => {
    recognizing = true;
    micBtn.innerHTML = `<img src="images/soundIcon.png" alt="Listening">`;
  };

  recognition.onresult = (event) => {
    let transcript = event.results[0][0].transcript;
    const corrected = correctTranscript(transcript);
    console.log(`🎙️ Heard: "${transcript}" → Corrected: "${corrected}"`);
    userInput.value = corrected;
    sendMessage();
  };

  recognition.onerror = () => {
    recognizing = false;
    micBtn.innerHTML = `<img src="images/micIcon.png" alt="Voice">`;
  };

  recognition.onend = () => {
    recognizing = false;
    micBtn.innerHTML = `<img src="images/micIcon.png" alt="Voice">`;
  };
}

micBtn.addEventListener("click", () => {
  try {
    if (!recognition) {
      alert("Voice recognition not supported in this browser.");
      return;
    }

    if (recognizing) {
      recognition.stop();
      micBtn.innerHTML = `<img src="images/micIcon.png" alt="Voice">`;
      recognizing = false;
    } else {
      recognition.start();
    }
  } catch (err) {
    console.error("🎤 Voice recognition error:", err);
    micBtn.innerHTML = `<img src="images/micIcon.png" alt="Voice">`;
    recognizing = false;
  }
});

// Enter key send
userInput.addEventListener("keypress", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    sendMessage();
  }
});

// Expand toggle
expandBtn.addEventListener("click", () => {
  chatbox.classList.toggle("expanded");
  expandBtn.innerHTML = chatbox.classList.contains("expanded") ? "🗗" : "⛶";
});

if (routeBtn) {
  routeBtn.addEventListener("click", computeRoute);
}

if (routeFeatureBtn && routePanel) {
  routeFeatureBtn.addEventListener("click", () => {
    routePanel.classList.toggle("hidden");
    routeFeatureBtn.textContent = routePanel.classList.contains("hidden") ? "🧭" : "🗺️";
  });
}

if (routeFrom) {
  routeFrom.addEventListener("change", filterDestinationOptions);
}

if (routeMapToggle) {
  routeMapToggle.addEventListener("click", () => {
    const isHidden = routeCanvasWrap?.classList.contains("hidden");
    if (isHidden && !hasDrawableRoute) {
      drawMapPlaceholder("Compute a route first to show the map preview.");
    }
    setMapVisibility(Boolean(isHidden));
  });
}
