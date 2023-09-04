import tkinter as tk
from tkinter import scrolledtext, messagebox
from internship import generate_summary

def get_screen_dimensions():
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    return screen_width, screen_height

window = tk.Tk()
window.title("Text Summarizer")

screen_width, screen_height = get_screen_dimensions()
window_width = int(0.9 * screen_width)
window_height = int(0.9 * screen_height)

window.geometry(f"{window_width}x{window_height}")

input_label = tk.Label(window, text="Enter your text:")
input_label.pack()

text_box = scrolledtext.ScrolledText(window, width=50, height=10)
text_box.pack()

method_label = tk.Label(window, text="Select Summarization Method:")
method_label.pack()

method_var = tk.StringVar(value="frequency")  
methods = ["frequency", "tfidf", "cosine"]
for method in methods:
    method_radio = tk.Radiobutton(window, text=method, variable=method_var, value=method)
    method_radio.pack()

ratio_label = tk.Label(window, text="Summary Ratio (0.1 - 1.0):")
ratio_label.pack()

ratio_entry = tk.Entry(window)
ratio_entry.pack()

def summarize_text():
    input_text = text_box.get("1.0", "end-1c") 
    method = method_var.get() 
    ratio = float(ratio_entry.get()) 
    try:
        summary = generate_summary(input_text, method=method, ratio=ratio)
        summary_box.delete("1.0", tk.END)  
        summary_box.insert(tk.END, summary) 
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

summarize_button = tk.Button(window, text="Summarize", command=summarize_text)
summarize_button.pack()

summary_label = tk.Label(window, text="Summary:")
summary_label.pack()

summary_box = scrolledtext.ScrolledText(window, width=50, height=10)
summary_box.pack()

window.mainloop()
