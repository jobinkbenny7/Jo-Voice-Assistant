import tkinter as tk
from tkinter import messagebox
import threading
import speech_recognition as sr
from jo import process_command

def on_submit():
    user_input = entry.get()
    if user_input.strip():
        output_text.insert(tk.END, f"You: {user_input}\n", "user")
        entry.delete(0, tk.END)
        threading.Thread(target=handle_command, args=(user_input,), daemon=True).start()

def handle_command(command):
    response = process_command(command)
    output_text.insert(tk.END, f"Jo: {response}\n", "jo")

def listen_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        output_text.insert(tk.END, "Listening...\n", "info")
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio)
            output_text.insert(tk.END, f"You: {command}\n", "user")
            threading.Thread(target=handle_command, args=(command,), daemon=True).start()
        except sr.UnknownValueError:
            output_text.insert(tk.END, "Could not understand audio\n", "error")
        except sr.RequestError:
            output_text.insert(tk.END, "Speech recognition service unavailable\n", "error")

def main():
    global entry, output_text
    root = tk.Tk()
    root.title("Jo - Interactive UI")
    root.geometry("400x500")
    root.configure(bg="#222")
    
    output_text = tk.Text(root, height=20, width=50, bg="#333", fg="white", font=("Arial", 12))
    output_text.pack(pady=10, padx=10)
    output_text.tag_config("user", foreground="#00ff00")
    output_text.tag_config("jo", foreground="#00bfff")
    output_text.tag_config("info", foreground="#ffff00")
    output_text.tag_config("error", foreground="#ff3333")
    
    entry = tk.Entry(root, width=40, font=("Arial", 12), bg="#444", fg="white", insertbackground="white")
    entry.pack(pady=5)
    
    button_frame = tk.Frame(root, bg="#222")
    button_frame.pack(pady=10)
    
    submit_btn = tk.Button(button_frame, text="Submit", command=on_submit, bg="#008000", fg="white", font=("Arial", 12), padx=10, pady=5)
    submit_btn.grid(row=0, column=0, padx=5)
    
    voice_btn = tk.Button(button_frame, text="Voice Input", command=listen_voice, bg="#ff8800", fg="white", font=("Arial", 12), padx=10, pady=5)
    voice_btn.grid(row=0, column=1, padx=5)
    
    root.mainloop()

if __name__ == "__main__":
    main()