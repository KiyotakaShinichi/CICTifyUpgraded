const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const micBtn = document.getElementById("mic-btn");
const chatbox = document.getElementById("chatbox");
const expandBtn = document.getElementById("expand-btn");

let abortController = null;
let recognizing = false;
let recognition;

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
    <div class="text">${message}</div>
  `;
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
    const botMsg = document.createElement("div");
    botMsg.classList.add("chat-message", "bot-message");
    botMsg.innerHTML = `
      <img src="images/CICTify_ChatLogo.png" alt="Bot Avatar">
      <div class="text">${botReply}</div>
    `;

    // 🔊 Listen button
    const listenBtn = document.createElement("button");
    listenBtn.textContent = "🔊 Listen";
    listenBtn.className = "tts-btn";
    listenBtn.style.marginTop = "5px";
    listenBtn.onclick = () => {
      const utterance = new SpeechSynthesisUtterance(botReply);
      utterance.rate = 1;
      utterance.pitch = 1;
      utterance.volume = 1;
      window.speechSynthesis.speak(utterance);
    };
    botMsg.appendChild(listenBtn);

    chatLog.appendChild(botMsg);
    chatLog.scrollTop = chatLog.scrollHeight;
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
