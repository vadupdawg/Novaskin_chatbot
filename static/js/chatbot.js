$("#input-message").on("keypress", function (e) {
  if (e.which == 13) {
    sendMessage();
  }
});

$("#send-btn").on("click", function () {
  sendMessage();
});

function getTimestamp() {
  const date = new Date();
  return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
}

async function sendChatRequest(userMessage, signal) {
  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_message: userMessage }),
    signal: signal,
  });
  return await response.json();
}

function handleChatbotResponse(response, timestamp) {
  let responseText = response.chatbot_response;
  $("#chat-output").append(`<div class="message bot"><strong>NovaSkin:</strong> ${responseText}<span class="timestamp-bot" style="float: left;">${timestamp}</span></div>`);
  $("#chat-output").scrollTop($("#chat-output")[0].scrollHeight);
}

function handleChatbotError(error, retryCount, userMessage) {
  console.error("Error:", error);
  if (error.name === 'AbortError') {
    if (retryCount > 0) {
      console.log(`Retrying... ${retryCount} attempts left.`);
      askChatBot(`${userMessage} maak het antwoord zo klein mogelijk`, retryCount - 1);
    } else {
      console.log('Failed after 3 attempts');
      $("#chat-output").append(`<div class="message bot"><strong>NovaSkin:</strong> Het antwoord op deze vraag is te lang, probeer de vraag kleiner te maken</div>`);
    }
  }
}

async function askChatBot(userMessage, retryCount = 3) {
  const controller = new AbortController();
  const signal = controller.signal;

  setTimeout(() => controller.abort(), 30000); // Dit annuleert de fetch request na 30 seconden

  try {
    $("#typing-indicator").show();
    startTypingAnimation();
    const response = await sendChatRequest(userMessage, signal);
    console.log("Response from the backend:", response);
    handleChatbotResponse(response, getTimestamp());
  } catch (error) {
    handleChatbotError(error, retryCount, userMessage);
  } finally {
    stopTypingAnimation();
  }
}

let typingAnimation;
function startTypingAnimation() {
  let dots = "";
  typingAnimation = setInterval(() => {
    if (dots.length < 3) {
      dots += ".";
    } else {
      dots = "";
    }
    $("#typing-indicator").text(`NovaSkin is typing${dots}`);
  }, 500);
}

function stopTypingAnimation() {
  clearInterval(typingAnimation);
  $("#typing-indicator").hide();
}

function sendMessage() {
  const userMessage = $("#input-message").val();
  if (userMessage.trim().length > 0) {
    const timestamp = getTimestamp();
    $("#chat-output").append(`<div class="message user"><strong>You:</strong> ${userMessage}<span class="timestamp-user" style="float: right;">${timestamp}</span></div>`);
    $("#input-message").val("");
    $("#char-counter").text("0/140 karakters"); // Deze lijn reset de teller
    var userInput = document.getElementById("input-message");
    userInput.disabled = true;
    setTimeout(function () {
      userInput.disabled = false;
      userInput.focus();
    }, 2000);

    // Scroll naar beneden na het versturen van een bericht
    $("#chat-output").scrollTop($("#chat-output")[0].scrollHeight);

    askChatBot(userMessage);
  }  
}


function changePlaceholder() {
  const inputField = $("#input-message");
  const placeholderText = inputField.attr("placeholder");

  if (placeholderText === "Type your message here") {
    inputField.attr("placeholder", "Typ uw bericht hier");
  } else {
    inputField.attr("placeholder", "Type your message here");
  }
}

  // Beperking van de tekstinvoerlengte en de tekensteller
  $("#input-message").attr("maxlength", 140);
  $("#input-message").on("input", function () {
    const currentLength = $(this).val().length;
    $("#char-counter").text(currentLength + "/140 karakters");
  });

setInterval(changePlaceholder, 5000);

$(document).ready(async function () {
  const timestamp = getTimestamp();
  $(".timestamp-greeting").text(timestamp);

  const response = await fetch("/start-chat", { method: "GET" });
  const jsonResponse = await response.json();
  const chatbotResponse = jsonResponse.chatbot_response;

  $("#chat-output").append(`<div class="message bot"><strong>NovaSkin:</strong> ${chatbotResponse}<span class="timestamp-bot" style="float: left;">${timestamp}</span></div>`);

});
