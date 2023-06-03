import tkinter as tk
from tkinter import *
import numpy as np
import joblib
import codecs
from sklearn.tree import DecisionTreeClassifier
import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import tkinter as tk
from tkinter import ttk
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
breed = 0 if 'Mix' in breed_entry.get() else 1
color = 1 if '/' in color_entry.get() else 0
dicti={'Dog': 0, 'Cat': 1, 'Intact Male': 1,'Intact Female': 0,'Neutered Male': 2, 'Spayed Female': 3, 'Unknown': 4,'0 years': 0,'1 year': 1,
       '2 years': 2,'3 years': 3,'4 years': 4,'5 years': 5,'6 years': 6,'7 years': 7,'8 years': 8,'9 years': 9,'10 years': 10,
       '11 years': 11,'12 years': 12,'13 years': 13,'14 years': 14,'15 years': 15,'16 years': 16,'17 years': 17,'18 years': 18,
       '19 years': 19,'20 years': 20,'1 month': 1/12,'2 months': 2/12,'3 months':3/12,'4 months':4/12,'5 months':5/12,'6 months':6/12,
       '7 months':7/12,'8 months': 8/12,'9 months': 9/12,'10 months': 10/12,'11 months': 11/12,'1 week':1/48,'1 weeks': 1/48,'2 weeks': 2/48,'3 weeks': 3/48,
       '4 weeks': 4/48,'5 weeks': 5/48,'1 day': 1/356,'2 days': 2/356,'3 days': 3/356,'4 days': 4/356,'5 days': 5/356,'6 days': 6/356,'nan' : 0}
# Create a function to predict using a trained model
def train_model():
    data = pd.read_csv("C:/Users/anvit/Downloads/train.csv")
    data = data.drop(['AnimalID', 'Name', 'OutcomeSubtype'], axis=1) 
    le = LabelEncoder()
    data['OutcomeType'] = le.fit_transform(data['OutcomeType'])
    # print(data.head())
    data_mod = data.drop('OutcomeType', axis=1)
    y = data['OutcomeType']
    res = pd.to_datetime(data_mod['DateTime'], errors='coerce')
    data_mod.Breed = [0 if 'Mix' in x else 1 for x in data_mod.Breed] 
    data_mod['Color'] = data_mod['Color'].apply(lambda x : 1 if '/' in x else 0)
    data_mod['AnimalType'] = [dicti[str(x)] for x in data_mod['AnimalType']]
    data_mod['Age'] = [dicti[str(x)] for x in data_mod['AgeuponOutcome']]
    # data_mod["Age"] = data_mod.apply(lambda row: label_age(row), axis=1)
    data_mod = data_mod.drop('AgeuponOutcome', axis=1)
    a = pd.to_datetime(data_mod['DateTime'])
    data_mod['month'] = a.dt.month
    data_mod['year'] = a.dt.year
    data_mod = data_mod.drop(['DateTime'], axis=1)
    data_mod = data_mod.fillna('missing')
    # data_mod["Age"] = data_mod.apply(lambda row: label_age(row), axis=1)
    data_mod['Age'].replace(['missing'], '1', inplace=True) 
    data_mod['SexuponOutcome'].replace(['missing'], 'Neutered Male', inplace=True) 
    data_mod['SexuponOutcome'] = [dicti[str(x)] for x in data_mod['SexuponOutcome']]
    
    print(data_mod)
    # print(data_mod.apply(lambda x: sum(x.isnull()/len(data_mod))))
    X_train, X_val, y_train, y_val = train_test_split(data_mod,y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=1000)
    model = rf.fit(X_train, y_train)      
    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val,y_val_pred)
    print(y_val_pred)
    print(model)
    # print("RF Accuracy: %.2f%%" % (accuracy * 100.0))
    # from sklearn.linear_model import LogisticRegression
    # lm = LogisticRegression()
    # lm = lm.fit(X_train,y_train)
    # #test the model
    # prediction = lm.predict(X_val)    
    # accuracy = accuracy_score(y_val,prediction)
    # print("LR Accuracy: %.2f%%" % (accuracy * 100.0))
    # from sklearn.neighbors import KNeighborsClassifier
    # knn = KNeighborsClassifier()
    # knn = knn.fit(X_train,y_train)
    # kn = knn.predict(X_val)
    # accknn = accuracy_score(kn,y_val)
    # print("KNN Accuracy: %.2f%%" % (accknn * 100.0))
    # from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    # qda = QuadraticDiscriminantAnalysis()
    # scores = cross_val_score(qda, X_train, y_train, cv=10)
    # accqda = scores.mean()*100
    # print(accqda)
    return  model

def predict_fun():
    features = [dicti[str(animal_entry.get())], dicti[str(sexcome_entry.get())], breed,
                color,dicti[str(age_entry.get())], month_entry.get(),year_entry.get()]
    X = np.array(features).reshape(1, -1)

    model = train_model()
    prediction=model.predict(X)   
    print(prediction) 
    # return "Transported" if prediction[0] == 1 else "Not Transported"

# Create a function to update the output label
# def update_output():
#     prediction = predict()
#     if(prediction[0] == 'T'):
#         output_label.config(text=f'Prediction: Transported')
#     else:
#         output_label.config(text=f'Prediction: Not Transported')

# # Create a button to trigger the prediction function
predict_button = tk.Button(root, text='Predict', font=('Comic Sans MS', 10, 'bold'))
predict_button.place(x=250, y=330)

# # Create an output label to display the predicted value
output_label = tk.Label(root, text='Prediction: ', font=('Comic Sans MS', 10, 'bold'))
output_label.place(x=250, y=430)

# #Decoded features
# homePlanet_decode1 = tk.Label(root, text='HomePlanet: ', font=('Comic Sans MS', 10, 'bold'))
# homePlanet_decode2 = tk.Label(root, text='[Europa, Earth, Mars]', font=('Comic Sans MS', 10, 'bold'))
# homePlanet_decode1.place(x=80, y=460)
# homePlanet_decode2.place(x=200, y=460)


# destination_decode1 = tk.Label(root, text='Destination: ', font=('Comic Sans MS', 10, 'bold'))
# destination_decode2 = tk.Label(root, text='[TRAPPIST-1e, PSO J318.5-22, 55 Cancri e]', font=('Comic Sans MS', 10, 'bold'))
# destination_decode1.place(x=80, y=490)
# destination_decode2.place(x=200, y=490)

# deck_decode1 = tk.Label(root, text='Cabin Deck: ', font=('Comic Sans MS', 10, 'bold'))
# deck_decode2 = tk.Label(root, text='[B, F, A, G, E, C, D]', font=('Comic Sans MS', 10, 'bold'))
# deck_decode1.place(x=80, y=520)
# deck_decode2.place(x=200, y=520)




root.mainloop()
