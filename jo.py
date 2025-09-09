import pyttsx3
import pywin32_system32
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser as wb
import os
import random
import pyautogui
import subprocess
import requests
import cv2
import numpy as np

engine = pyttsx3.init()

def speak(audio) -> None:
    engine.say(audio)
    engine.runAndWait()

def time() -> None:
    Time = datetime.datetime.now().strftime("%I:%M:%S")
    speak("the current time is")
    speak(Time)
    print("The current time is ", Time)

def date() -> None:
    day: int = datetime.datetime.now().day
    month: int = datetime.datetime.now().month
    year: int = datetime.datetime.now().year
    speak("the current date is")
    speak(day)
    speak(month)
    speak(year)
    print(f"The current date is {day}/{month}/{year}")

def wishme() -> None:
    print("Welcome back sir!!")
    speak("Welcome back sir!!")

    hour: int = datetime.datetime.now().hour
    if 4 <= hour < 12:
        speak("Good Morning !!")
        print("Good Morning !!")
    elif 12 <= hour < 16:
        speak("Good Afternoon !!")
        print("Good Afternoon !!")
    elif 16 <= hour < 24:
        speak("Good Evening !!")
        print("Good Evening !!")
    else:
        speak("Good Night , See You Tomorrow")

    speak("Jo at your service, what is your command")
    print("Jo at your service, what is your command")

def screenshot() -> None:
    img = pyautogui.screenshot()
    img_path = os.path.expanduser("~\\Pictures\\Screenshots.png")
    img.save(img_path)

def tell_joke():
    try:
        response = requests.get("https://official-joke-api.appspot.com/random_joke", timeout=5)
        response.raise_for_status()  # Raises an exception for HTTP errors
        joke_data = response.json()
        joke = f"{joke_data['setup']} ... {joke_data['punchline']}"
        speak(joke) 
        print(joke)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching joke: {e}")
        speak("Sorry, I couldn't fetch a joke right now. Here's one: Why don't scientists trust atoms? Because they make up everything!")

def get_news():
    try:
        api_key = "7bfc0c16017e4e3f83b5bcdbf6b17b5e"  # Consider moving this to a config file
        news_url = f"https://newsapi.org/v2/top-headlines?country=us&language=en&apiKey={api_key}"
        
        response = requests.get(news_url, timeout=5)
        response.raise_for_status()
        news_data = response.json()
        
        if news_data.get('status') == 'ok' and news_data.get('totalResults', 0) > 0:
            articles = news_data.get('articles', [])[:5]
            headlines = [article.get('title', 'No title') for article in articles]
            
            news_report = "Here are the top news headlines:\n"
            news_report += "\n".join([f"{i+1}. {headline}" for i, headline in enumerate(headlines) if headline])
            
            speak(news_report)
            print(news_report)
        else:
            speak("Sorry, I couldn't find any news articles at the moment.")
    
    except requests.exceptions.RequestException as e:
        print(f"News API Error: {e}")
        speak("Sorry, I couldn't retrieve the news right now.")
    
    except Exception as e:
        speak("Sorry, I couldn't retrieve the news right now.")
        print(f"Error: {e}")

def search_on_chrome():
    try:
        speak("What do you want to search?")
        print("What do you want to search?")
        search_query = take_command().lower()
        if search_query == "Try Again":
            speak("I didn't catch that. Please repeat.")
            return
        search_query = search_query.replace(" ", "+")
        chrome_path = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        if os.path.exists(chrome_path):
            search_url = f"https://www.google.com/search?q={search_query}"
            subprocess.Popen([chrome_path, search_url])
            
            speak(f"Searching for {search_query} on Chrome")
            print(f"Searching for {search_query} on Chrome")
        else:
            speak("Chrome browser is not installed or the path is incorrect.")
            print("Chrome browser is not installed or the path is incorrect.")
    
    except Exception as e:
        speak("Unable to open Chrome, please try again later.")
        print(f"Error occurred: {e}")

def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("I'm Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Processing command...")
        query = r.recognize_google(audio, language="en-in")
        print(query)

    except Exception as e:
        print(e)
        speak("I did not get that, please repeat")
        return "Try Again"

    return query

def get_number(prompt):
    while True:
        speak(prompt)
        num = take_command()
        
        if num == "Try Again":
            speak("I didn't catch that. Please say the number again.")
            continue
        
        try:
            return float(num)
        except ValueError:
            speak("That doesn't seem like a number. Please try again.")
            print(f"Invalid input: {num}. Please try again.")

def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Could not open camera.")
        print("Could not open camera.")
        return

    speak("Opening the camera. Press 'q' to exit.")
    print("Opening the camera. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            speak("Failed to capture image.")
            print("Failed to capture image.")
            break

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def record_screen():
    screen_size = (1920, 1080) 
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_file = os.path.expanduser("~\\Videos\\screen_recording.avi")
    out = cv2.VideoWriter(output_file, fourcc, 20.0, screen_size)

    speak("Recording the screen. Press 'q' to stop.")
    print("Recording the screen. Press 'q' to stop.")

    while True:
        img = pyautogui.screenshot() 
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
        out.write(frame) 

        cv2.imshow("Screen Recording", frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()
    cv2.destroyAllWindows()

def calculator():
    speak("What operation would you like to perform? You can say addition, subtraction, multiplication, or division.")
    print("What operation would you like to perform? You can say addition, subtraction, multiplication, or division.")
    
    operation = take_command().lower()
    
    if "add" in operation or "addition" in operation or "plus" in operation:
        num1 = get_number("Please say the first number")
        num2 = get_number("Please say the second number")
        result = num1 + num2
        speak(f"The result of addition is {result}")
        print(f"The result of addition is {result}")

    elif "subtract" in operation or "subtraction" in operation or "minus" in operation:
        num1 = get_number("Please say the first number")
        num2 = get_number("Please say the second number")
        result = num1 - num2
        speak(f"The result of subtraction is {result}")
        print(f"The result of subtraction is {result}")

    elif "multiply" in operation or "multiplication" in operation or "times" in operation:
        num1 = get_number("Please say the first number")
        num2 = get_number("Please say the second number")
        result = num1 * num2
        speak(f"The result of multiplication is {result}")
        print(f"The result of multiplication is {result}")

    elif "divide" in operation or "division" in operation or "over" in operation:
        num1 = get_number("Please say the first number")
        num2 = get_number("Please say the second number")
        if num2 != 0:
            result = num1 / num2
            speak(f"The result of division is {result}")
            print(f"The result of division is {result}")
        else:
            speak("Division by zero is not allowed.")
            print("Division by zero is not allowed.")

    else:
        speak("I did not understand the operation.")
        print("Invalid operation. Please try again.")

def clear_terminal() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')

def contains_word(s, word):
    return f' {word} ' in f' {s} '

def process_command(query):
    query = query.lower()
    
    # Time commands
    if contains_word(query, 'time') or contains_word(query, 'clock'):
        time()
    
    # Date commands
    elif contains_word(query, 'date') or contains_word(query, 'day') or contains_word(query, 'today'):
        date()
    
    # Introduction commands
    elif contains_word(query, 'who are you') or contains_word(query, 'your name'):
        speak("I'm Jo, your personal voice assistant.")
        print("I'm Jo, your personal voice assistant.")
    
    # Greeting commands
    elif contains_word(query, 'how are you'):
        speak("I'm good, how are you sir?")
        print("I'm good, how are you sir?")
    
    # Response to greetings
    elif contains_word(query, 'fine') or contains_word(query, 'good'):
        speak("Pleased to hear that sir!!")
        print("Pleased to hear that sir!!")
    
    # Wikipedia commands
    elif contains_word(query, 'wikipedia') or contains_word(query, 'search') or contains_word(query, 'look up'):
        try:
            speak("kindly wait, loading results...")
            query = query.replace("wikipedia", "")
            result = wikipedia.summary(query, sentences=2)
            print(result)
            speak(result)
        except:
            speak("Can't find this page sir, please ask something else")
    
    # Website commands
    elif contains_word(query, 'youtube'):
        wb.open("youtube.com")
    
    elif contains_word(query, 'google'):
        wb.open("google.com")
    
    # Calculator commands
    elif contains_word(query, 'calculate') or contains_word(query, 'calculator') or contains_word(query, 'math'):
        calculator()
    
    # College website
    elif contains_word(query, 'college') or contains_word(query, 'kristu jayanti'):
        wb.open("kristujayanti.edu.in")
    
    # Joke commands
    elif contains_word(query, 'joke') or contains_word(query, 'funny'):
        tell_joke()
    
    # News commands
    elif contains_word(query, 'news') or contains_word(query, 'headlines'):
        get_news()
    
    # Chrome search commands
    elif contains_word(query, 'search') or contains_word(query, 'chrome'):
        search_on_chrome()
    
    # Camera commands
elif contains_word(query, 'camera') or contains_word(query, 'photo'):
    open_camera()

# Music commands
elif contains_word(query, 'music') or contains_word(query, 'song') or contains_word(query, 'play'):
    song_dir = os.path.expanduser("C:\\Users\\jobin\\Desktop\\Jo-Voice-Assistant\\music")
    songs = os.listdir(song_dir)
    if songs:
        print(songs)
        x = len(songs) - 1
        y = secrets.randbelow(x)
        os.startfile(os.path.join(song_dir, songs[y]))
    else:
        speak("Your music folder is empty.")
        print("No songs found in the music directory.")

# Screenshot commands
elif contains_word(query, 'screenshot') or contains_word(query, 'capture'):
    screenshot()
    speak("I've taken a screenshot, please check it")
    # Screen recording commands
    elif contains_word(query, 'record') or contains_word(query, 'recording'):
        record_screen()
        speak("I've recorded the video, please check it")
    
    # Exit commands
    elif contains_word(query, 'offline') or contains_word(query, 'exit') or contains_word(query, 'quit'):
        speak("Goodbye sir! Have a great day.")
        print("Goodbye sir! Have a great day.")
        quit()
    
    # Clear terminal commands
    elif contains_word(query, 'clear') or contains_word(query, 'clean'):
        clear_terminal()
    
    else:
        speak("I didn't understand that command. Please try again.")
        print("Command not recognized.")

def main():
    wishme()
    while True:
        query = take_command().lower()
        process_command(query)

if __name__ == "__main__":
    main()