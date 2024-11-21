import pandas as pd
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tkinter import messagebox

# manipulate the data set
dataset = pd.read_csv("mail_data.csv")

# replace all the nulls with an empty string/remove the nulls from the data set
data = dataset.where(pd.notnull(dataset), "")


# change the category labels, ham to 1 and spam to 0
data.loc[data["Category"] == "spam", "Category"] = 0
data.loc[data["Category"] == "ham", "Category"] = 1

X = data["Message"]  # assign the messages to X
Y = data["Category"]  # assign the the categories (0,1) to Y

# split each set (message & category) into two subsets
# the train set and the test set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=3
)

# an instance for tranforming data into numerical features
# min_df=1 is the minimum number of lines a word should appear in order to include it in the features
# stop_words='english' removes common english words
# covvert all words to lower cases
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

# transform the X train and test
X_train_feature = feature_extraction.fit_transform(X_train)
X_test_feature = feature_extraction.transform(X_test)

# convert the Y train and test to integers
Y_train = Y_train.astype("int")
Y_test = Y_test.astype("int")

model = LogisticRegression()

# fit the train feature and Y train set into logistic regression
model.fit(X_train_feature, Y_train)

# predict the train set based on train feature
prediction_on_training_data = model.predict(X_train_feature)

# find the accuracy between predicted set and the actual set
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

# predict the test set based on test feature
prediction_on_testing_data = model.predict(X_test_feature)

# find the accuracy between predicted set and the actual set
accuracy_on_testing_data = accuracy_score(Y_test, prediction_on_testing_data)

# graphical user interface main window
root = tk.Tk()
root.title("Spam Detector")


# message for detection results
def results():
    user_text = message_txtbox.get().lower()
    if user_text:
        try:
            input_your_mail = [user_text]
            input_data_feature = feature_extraction.transform(input_your_mail)
            prediction = model.predict(input_data_feature)
            if prediction[0] == 1:
                messagebox.showinfo("Message", "The entered message is not a Spam.")
            else:
                messagebox.showinfo("Message", "The entered message is Spam.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    else:
        messagebox.showerror("Error", "Please enter your message")


# prompt message label
prompt_message = tk.Label(root, text="Enter your email or message:")
prompt_message.pack(pady=10)

# text box
message_txtbox = tk.Entry(root, width=40)
message_txtbox.pack(pady=10)

# button
detect_button = tk.Button(root, text="Click to Detect", command=results)
detect_button.pack(pady=50, padx=100)


# Run the application
root.mainloop()
