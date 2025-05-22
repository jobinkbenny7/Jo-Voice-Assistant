// Maps commands to Python code snippets
const commandTemplates = {
    time: "time()",
    date: "date()",
    camera: "open_camera()",
    joke: "tell_joke()"
};

function generateCode(command) {
    const code = `# Generated command\n${commandTemplates[command]}`;
    document.getElementById("generatedCode").innerText = code;
    updateConsole(`Generated code for: ${command}`);
}

function copyCode() {
    const code = document.getElementById("generatedCode").innerText;
    navigator.clipboard.writeText(code);
    updateConsole("Code copied to clipboard!");
}

function executeCode() {
    const code = document.getElementById("generatedCode").innerText;
    fetch('/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: code })
    })
    .then(response => response.json())
    .then(data => updateConsole(data.output));
}

function updateConsole(message) {
    const consoleDiv = document.getElementById("consoleOutput");
    consoleDiv.innerHTML += `<div>${new Date().toLocaleTimeString()}: ${message}</div>`;
}