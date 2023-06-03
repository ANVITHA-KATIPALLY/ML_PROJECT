import tkinter as tk
from tkinter import *
import numpy as np
import codecs
from sklearn.tree import DecisionTreeClassifier
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageTk

# Create a tkinter window
root = tk.Tk()
root.geometry('600x600')
# Load the image using Pillow
pil_image = Image.open("C:/Users/anvit/OneDrive/Desktop/maxresdefault.jpg")

# Convert the Pillow image to a Tkinter PhotoImage
tk_image = ImageTk.PhotoImage(pil_image)

# Display the image in a label
img_label = tk.Label(root, image=tk_image)
img_label.pack()
# Create input labels and entry boxes

animal_label = tk.Label(root, text='Animal: ', font=('Comic Sans MS', 10, 'bold'))
animal_entry = tk.Entry(root)
animal_label.place(x=150, y=10)
animal_entry.place(x=350, y=10)

sexcome_label = tk.Label(root, text='SexType:', font=('Comic Sans MS', 10, 'bold'))
sexcome_label.place(x=150, y=40)
sexcome_entry = tk.Entry(root)
sexcome_entry.place(x=350, y=40)

breed_label = tk.Label(root, text='breed:', font=('Comic Sans MS', 10, 'bold'))
breed_label.place(x=150, y=70)
breed_entry = tk.Entry(root)
breed_entry.place(x=350, y=70)

color_label = tk.Label(root, text='Color:', font=('Comic Sans MS', 10, 'bold'))
color_label.place(x=150, y=100)
color_entry = tk.Entry(root)
color_entry.place(x=350, y=100)

age_label = tk.Label(root, text='Age:', font=('Comic Sans MS', 10, 'bold'))
age_label.place(x=150, y=130)
age_entry = tk.Entry(root)
age_entry.place(x=350, y=130)

month_label = tk.Label(root, text='Month:', font=('Comic Sans MS', 10, 'bold'))
month_label.place(x=150, y=160)
month_entry = tk.Entry(root)
month_entry.place(x=350, y=160)

year_label = tk.Label(root, text='Year:', font=('Comic Sans MS', 10, 'bold'))
year_label.place(x=150, y=190)
year_entry = tk.Entry(root)
year_entry.place(x=350, y=190)

# # Create a button to trigger the prediction function
predict_button = tk.Button(root, text='Predict', font=('Comic Sans MS', 10, 'bold'))
predict_button.place(x=250, y=330)

# # Create an output label to display the predicted value
output_label = tk.Label(root, text='Prediction: ', font=('Comic Sans MS', 10, 'bold'))
output_label.place(x=250, y=430)

root.mainloop()
