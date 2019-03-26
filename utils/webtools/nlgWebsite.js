var inpF, hist;
var inpFdefault = "Enter your text here.";

window.onload = setup;

function newConversation(){
  // clear out the history and input field
  hist = document.getElementById("history");
  hist.innerHTML = "";
  inpF.value = inpFdefault;

  goal = document.getElementById("goal");
  var xmlHttp = new XMLHttpRequest();
  xmlHttp.onreadystatechange = function() {
    if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
      new_goal = xmlHttp.responseText;
      goal.innerHTML = new_goal;
      console.log(new_goal);
    }
  }
  xmlHttp.open("GET", window.location.href+"goal", true);
  xmlHttp.send();
}

function getResponse(){
  // Pushing prompt to the history
  var prompt = inpF.value;
  if (prompt === inpFdefault || prompt === "")
    return; // show warning message?
  hist = document.getElementById("history");
  hist.innerHTML += `<div class="history chat user">${prompt}</div>`;
  inpF.value = "";
  console.log(inpF.value);
  console.log(prompt);

  // Sending stuff to server (for now, no server)
  var nlgResponse = "Default Text";
  var toSend = $.trim(prompt).replace(/\n/g,"|||");
  // var toSend = $.trim(hist.innerText).replace(/\n/g,"|||");
  var xmlHttp = new XMLHttpRequest();
  xmlHttp.onreadystatechange = function() {
    if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
      console.log(xmlHttp.responseText);
      nlgResponse = xmlHttp.responseText;
      hist.innerHTML += `<div class="history chat agent dialogue">${nlgResponse}</div>`;
    }
  }
  xmlHttp.open("POST", window.location.href+"?inputText="+toSend, true);
  xmlHttp.send();
}

function setup(){
  /*Input field aesthetics*/
  inpF = document.getElementById("inputfield");
  inpF.onfocus = function(){
    if(inpF.value === inpFdefault){
      inpF.style = "color: black;font-style: normal";
      inpF.value = "";
    }
  };
  inpF.onblur = function(){
    if (inpF.value === ""){
      inpF.style = "color: grey;font-style: italic;";
      inpF.value = inpFdefault;
    }
  };
  inpF.onkeyup = function(e){
    if (e.keyCode === 13 && !e.shiftKey){
      getResponse();
    }
  };
  newConversation();
}