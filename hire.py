from tkinter import *
import pandas as pd
import numpy as np
from patsy.highlevel import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xlsxwriter

def submit():
    r1 = lr.get()
    r2 = dt.get()
    r3 = rf.get()
    r4 = sv.get()
    p = int(percent.get())
    b = int(backlog.get())
    i = int(intern.get())
    f = int(first.get())
    c = int(comm.get())
    results = [r1, r2, r3, r4]
    if results.count('Hire') > results.count('Not Hire'):
        hire.set('Hire')
        h = 1
    else:
        hire.set('Not Hire')
        h = 0

    input_file = 'dataset.xlsx'
    sheet = 'Sheet1'
    df = pd.read_excel(input_file, sheet, header=0)
    features = list(df.columns[:6])
    new_df = pd.DataFrame([[p, b, i, f, c, h]], columns=features)
    df = df.append(new_df, ignore_index=True)
    writer = pd.ExcelWriter(input_file, engine='xlsxwriter')
    df.to_excel(writer, sheet, index=False)
    writer.save()
def LogReg():
    n = name.get()
    p = int(percent.get())
    b = int(backlog.get())
    i = int(intern.get())
    f = int(first.get())
    c = int(comm.get())
    input_file = 'dataSet.xlsx'
    sheet = 'Sheet1'
    df = pd.read_excel(input_file, sheet, header=0)
    y, X = dmatrices('Hire ~ Percentage + Backlog + Internship + First_Round + Communication_Skills', df,
                     return_type='dataframe')
    y = np.ravel(y)
    model = LogisticRegression()
    model = model.fit(X, y)
    h = int(model.predict(np.array([1, p, b, i, f, c]).reshape(1, -1)))
    if h == 1:
        lr.set("Hire")
    else:
        lr.set("Not Hire")


def dtree():
    input_file = 'dataSet.xlsx'
    sheet = 'Sheet1'
    df = pd.read_excel(input_file, sheet, header=0)
    n = name.get()
    p = int(percent.get())
    b = int(backlog.get())
    i = int(intern.get())
    f = int(first.get())
    c = int(comm.get())
    features = list(df.columns[:5])
    y = df['Hire']
    X = df[features]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    h = int(clf.predict(np.array([p, b, i, f, c]).reshape(1, -1)))
    if h == 1:
        dt.set("Hire")
    else:
        dt.set("Not Hire")


def forest():
    input_file = 'dataSet.xlsx'
    sheet = 'Sheet1'
    df = pd.read_excel(input_file, sheet, header=0)
    n = name.get()
    p = int(percent.get())
    b = int(backlog.get())
    i = int(intern.get())
    f = int(first.get())
    c = int(comm.get())
    features = list(df.columns[:5])
    y = df['Hire']
    X = df[features]

    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X, y)
    h = int(clf.predict([[p, b, i, f, c]]))
    if h == 1:
        rf.set("Hire")
    else:
        rf.set("Not Hire")


def svm_result():
    input_file = 'dataSet.xlsx'
    sheet = 'Sheet1'

    n = name.get()
    p = int(percent.get())
    b = int(backlog.get())
    i = int(intern.get())
    f = int(first.get())
    c = int(comm.get())
    df = pd.read_excel(input_file, sheet, header=0)
    features = list(df.columns[:5])
    y = df['Hire']
    X = df[features]

    C = 2.0  # smoothness value
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    h = svc.predict([[p, b, i, f, c]])
    if h == 1:
        sv.set("Hire")
    else:
        sv.set("Not Hire")


win = Tk()
win.geometry("1000x760")
win.title("window application")
win.configure(bg='white')

frame = Frame(win, width=1000, height=80, bg='Red', bd=10, relief='raised')
frame.pack(side='top')

Label(frame, text='Hire Predictor', font=('times new roman', 50, 'bold'), bg='light grey', fg='black', width=24).grid(
    row=0, column=0)

frame1 = Frame(win, width=1000, height=1000, bg='brown', bd=10, relief='raised')
frame1.pack(padx=20, pady=20)

lb = Label(frame1, text='Name:', font=('times new roman', 15), bg='grey', fg='black', width=13)
lb.grid(row=0, column=0, padx=10, pady=10)

lb1 = Label(frame1, text='Percentage:', font=('times new roman', 15), bg='grey', fg='black', width=13)
lb1.grid(row=1, column=0, padx=10)

lb2 = Label(frame1, text='Backlogs:', font=('times new roman', 15), bg='grey', fg='black', width=13)
lb2.grid(row=2, column=0, padx=10)
lb3 = Label(frame1, text='Internship:', font=('times new roman', 15,), bg='grey', fg='black', width=13)
lb3.grid(row=3, column=0, padx=10)
lb4 = Label(frame1, text='FirstRound:', font=('times new roman', 15), bg='grey', fg='black', width=13)
lb4.grid(row=4, column=0, padx=10)
lb5 = Label(frame1, text='Communication:', font=('times new roman', 15), bg='grey', fg='black', width=13)
lb5.grid(row=5, column=0, padx=10, pady=10)
lb11 = Label(frame1, text='Hire: ', font=('times new roman', 15), bg='grey', fg='black', width=13)
lb11.grid(row=6, column=0, padx=10, pady=10)

lb6 = Label(frame1, text='Logistic Result', font=('times new roman', 12, 'bold'), bg='grey', fg='black', width=18)
lb6.grid(row=7, column=0, pady=15)
lb7 = Label(frame1, text='Decision Tree Result', font=('times new roman', 12, 'bold'), bg='grey', fg='black', width=18)
lb7.grid(row=7, column=1, padx=10, pady=10)
lb8 = Label(frame1, text='Random Forest Result', font=('times new roman', 12, 'bold'), bg='grey', fg='black', width=18)
lb8.grid(row=7, column=2, padx=10, pady=10)
lb9 = Label(frame1, text='SVM Result', font=('times new roman', 12, 'bold'), bg='grey', fg='black', width=18)
lb9.grid(row=7, column=3, padx=10, pady=10)

name = StringVar()
tb1 = Entry(frame1, textvariable=name)
tb1.grid(row=0, column=1, ipadx=15, ipady=2)

percent = StringVar()
tb2 = Entry(frame1, textvariable=percent)
tb2.grid(row=1, column=1, ipadx=15, ipady=2)

backlog = StringVar()
tb3 = Entry(frame1, textvariable=backlog)
tb3.grid(row=2, column=1, ipadx=15, ipady=2)

intern = StringVar()
tb4 = Entry(frame1, textvariable=intern)
tb4.grid(row=3, column=1, ipadx=15, ipady=2)

first = StringVar()
tb5 = Entry(frame1, textvariable=first)
tb5.grid(row=4, column=1, ipadx=15, ipady=2)

comm = StringVar()
tb6 = Entry(frame1, textvariable=comm)
tb6.grid(row=5, column=1, ipadx=15, ipady=2)

hire = StringVar()
tb = Entry(frame1, textvariable=hire, state='disabled',font=('times new roman', 15, 'bold'),fg='brown')
tb.grid(row=6, column=1, ipadx=15, ipady=2)

lr = StringVar()
dt = StringVar()
rf = StringVar()
sv = StringVar()

tb7 = Entry(frame1, textvariable=lr, state='disabled',font=('times new roman', 15, 'bold'), fg='black')
tb7.grid(row=8, column=0, padx=5, pady=5, ipadx=10, ipady=10)

tb8 = Entry(frame1, textvariable=dt, state='disabled',font=('times new roman', 15, 'bold'), fg='black')
tb8.grid(row=8, column=1, padx=5, pady=5, ipadx=10, ipady=10)

tb9 = Entry(frame1, textvariable=rf, state='disabled',font=('times new roman', 15, 'bold'), fg='black')
tb9.grid(row=8, column=2, padx=5, pady=5, ipadx=10, ipady=10)

tb10 = Entry(frame1, textvariable=sv, state='disabled',font=('times new roman', 15, 'bold'), fg='black')
tb10.grid(row=8, column=3, padx=5, pady=5, ipadx=10, ipady=10)

btn1 = Button(frame1, text="Logistic", font=("times new roman", 15, 'bold'), bg='grey', fg='black', width=13,
              command=LogReg)
btn1.grid(row=1, column=3, pady=10)
btn2 = Button(frame1, text="Decision Tree", font=("times new roman", 15, 'bold'), bg='grey', fg='black', width=13,
              command=dtree)
btn2.grid(row=2, column=3, pady=10)
btn3 = Button(frame1, text="Random Forest", font=("times new roman", 15, 'bold'), bg='grey', fg='black', width=13,
              command=forest)
btn3.grid(row=3, column=3, pady=10)
btn4 = Button(frame1, text="SVM", font=("times new roman", 15, 'bold'), bg='grey', fg='black', width=13,
              command=svm_result)
btn4.grid(row=4, column=3, pady=10)

btn5 = Button(frame1, text="Final Result", font=("times new roman", 18), bg='grey', fg='black', width=13,
              command=submit)
btn5.grid(row=9, columnspan=7, pady=15, padx=20)

win.mainloop()