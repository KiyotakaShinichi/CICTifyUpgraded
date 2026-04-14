(() => {
  const chatBtn = document.createElement("img");
  const chatBox = document.createElement("div");
  const chatFrame = document.createElement("iframe");

  // Floating button (with subtle pulse effect)
  Object.assign(chatBtn.style, {
    position: "fixed",
    bottom: "20px",
    right: "20px",
    zIndex: "9999",
    width: "70px",
    height: "70px",
    borderRadius: "50%",
    cursor: "pointer",
    boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
    transition: "transform 0.25s ease",
    animation: "pulse 2s infinite ease-in-out",
  });
  chatBtn.src = "https://cictify.onrender.com/images/floatingCICTify.png";
  chatBtn.alt = "Open CICTify Chat";

  // Chat box
  Object.assign(chatBox.style, {
    position: "fixed",
    bottom: "100px",
    right: "20px",
    width: "380px",
    height: "500px",
    background: "#fff",
    borderRadius: "16px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
    overflow: "hidden",
    zIndex: "9999",
    display: "none",
    flexDirection: "column",
  });

  // Embedded chatbot (Render URL)
  chatFrame.src = "https://cictify.onrender.com";
  Object.assign(chatFrame.style, {
    width: "100%",
    height: "100%",
    border: "none",
  });

  chatBox.appendChild(chatFrame);
  document.body.appendChild(chatBox);
  document.body.appendChild(chatBtn);

  // Toggle open/close
  chatBtn.addEventListener("click", () => {
    const open = chatBox.style.display === "none";
    chatBox.style.display = open ? "flex" : "none";
    chatBtn.style.transform = open ? "scale(0.9)" : "scale(1)";
  });

  // Inject CSS keyframes dynamically
  const style = document.createElement("style");
  style.textContent = `
    @keyframes pulse {
      0%, 100% { transform: scale(1); box-shadow: 0 0 0 rgba(255,115,0,0.4); }
      50% { transform: scale(1.05); box-shadow: 0 0 15px rgba(255,115,0,0.6); }
    }
  `;
  document.head.appendChild(style);
})();
