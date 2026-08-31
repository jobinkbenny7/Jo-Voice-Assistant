# Jo Voice Assistant 🎙️

Jo Voice Assistant is a feature-rich, Python-based voice assistant designed to help users automate everyday tasks through voice commands and interactive interfaces. Whether you want to fetch the latest news, perform mathematical calculations, take a screenshot, or just hear a joke, Jo is here to assist you!

## 🚀 Features

Jo comes packed with a variety of hands-free capabilities:

- **Time and Date:** Get accurate current time and date updates.
- **Web Browsing & Search:** Search Wikipedia, perform searches on Google Chrome, or instantly open websites like YouTube or Kristu Jayanti College.
- **Entertainment:** Ask Jo to tell a joke or play local music from your computer.
- **News Updates:** Fetch the top global news headlines using the NewsAPI.
- **Math Calculator:** Perform addition, subtraction, multiplication, and division entirely via voice.
- **Media & Screen Control:** 
  - Access and open your computer's web camera.
  - Take quick screenshots.
  - Record your screen effortlessly.
- **System Commands:** Clear the terminal or exit the program seamlessly.

## 💻 Interfaces Available

Jo isn't just limited to the terminal! The project includes multiple ways to interact:

1. **Terminal / Command Line:** Run `jo.py` for a fully hands-free voice experience directly in your terminal.
2. **Desktop GUI:** Run `Ui.py` for a sleek Tkinter-based desktop application that supports both typed text and voice commands.
3. **Web Application:** Run `app.py` to start a Flask web server and interact with Jo via a web browser.

## 🛠️ Installation & Setup

### Prerequisites
Ensure you have Python 3.x installed on your system.

### 1. Navigate to the Directory
If you have downloaded the repository, navigate into the project folder:
```bash
cd Jo-Voice-Assistant-main
```

### 2. Install Dependencies
Jo relies on a few external libraries. Install all required dependencies using `pip`:
```bash
pip install pyttsx3 SpeechRecognition wikipedia pyautogui requests opencv-python numpy Flask pywin32
```
*Note: `PyAudio` is also required for SpeechRecognition to access your microphone. If you are on Windows, you can install it via `pip install PyAudio`.*

### 3. API Configuration
- **NewsAPI:** Jo fetches news using NewsAPI. An API key is currently hardcoded, but for heavy usage, it is recommended to get your own key from [NewsAPI](https://newsapi.org/) and replace it inside the `get_news()` function in `jo.py`.
- **Paths:** Be sure to adjust the `song_dir` path in the `music` command section of `jo.py` to match the path of your local music folder.

## 🏃‍♂️ How to Run

Depending on the interface you prefer, execute one of the following commands in your terminal:

**For the Terminal Voice Assistant:**
```bash
python jo.py
```

**For the Desktop Graphical Interface (GUI):**
```bash
python Ui.py
```

**For the Web Interface:**
```bash
python app.py
```
*(After running, navigate to `http://127.0.0.1:5000` in your web browser)*

## 🗣️ Example Commands
Try saying any of the following phrases to Jo:
- *"What time is it?"*
- *"Search Wikipedia for Python"*
- *"Tell me a joke"*
- *"What's the latest news?"*
- *"Calculate addition"* (Jo will then prompt for the numbers)
- *"Take a screenshot"*
- *"Record screen"*
- *"Play music"*
- *"Open YouTube"*
- *"Who are you?"*

## 🤝 Contributing
Contributions, issues, and feature requests are highly welcome! Feel free to improve the code, add new voice commands, or enhance the graphical interfaces.

---
*Built in Python to offer convenience and accessibility with hands-free functionality.*
