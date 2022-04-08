import tkinter.messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import pandas as pd
import numpy as np
from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tkinter import filedialog
from sklearn.preprocessing import MinMaxScaler

# Đọc dữ liệu để huấn luyện
df = pd.read_csv('data.csv')
name_column = df.drop(columns=['diagnosis', 'id', 'Unnamed: 32'], axis=1).columns
df.drop(['id','Unnamed: 32'], axis=1, inplace=True)
knn = KNeighborsClassifier(n_neighbors=10)
scaler = MinMaxScaler()

def ML(df):
    corr_matrix = df.corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
    tri_df = corr_matrix.mask(mask)
    # Xóa những thuộc tính có độ tương quan cao
    global to_drop
    to_drop = [x for x in tri_df.columns if any(tri_df[x] > 0.9)]
    df = df.drop(to_drop, axis = 1)

    y = df['diagnosis']
    X = df.drop('diagnosis',axis=1)

    # chia tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # huấn luyện
    knn.fit(X_train, y_train)

def VisualScreen(index_col):
    name_col = name_column[index_col]
    visualScreen = Tk()
    visualScreen.title('Phân tích dữ liệu')
    visualScreen.geometry('900x700')
    Label(visualScreen, text='THÔNG TIN THUỘC TÍNH', font=('bold', 25)).place(x=250, y=20)

    # Thông tin thuộc tính: MEAN, STD, VAR, Min, Max,...
    Label(visualScreen, text='MEAN: '+str(round(df[name_col].mean(), 3)), font=('bold', 15)).place(x=120, y=450)
    Label(visualScreen, text='STANDARD DEVIATION: '+str(round(df[name_col].std(),3)), font=('bold', 15)).place(x=120, y=480)
    Label(visualScreen, text='VARIANCE: '+str(round(df[name_col].var(), 3)), font=('bold', 15)).place(x=120, y=510)

    cal_values = np.percentile(df[name_col], q=[0, 25, 50, 75, 100])

    Label(visualScreen, text='MIN: '+str(round(cal_values[0], 3)), font=('bold', 15)).place(x=550, y=450)
    Label(visualScreen, text='25%: '+str(round(cal_values[1], 3)), font=('bold', 15)).place(x=550, y=480)
    Label(visualScreen, text='50%: '+str(round(cal_values[2], 3)), font=('bold', 15)).place(x=550, y=510)
    Label(visualScreen, text='75%: '+str(round(cal_values[3], 3)), font=('bold', 15)).place(x=550, y=540)
    Label(visualScreen, text='MAX: '+str(round(cal_values[4], 3)), font=('bold', 15)).place(x=550, y=570)

    # BOXPLOT của thuộc tính được chọn
    figure1 = plt.Figure(figsize=(5, 4), dpi=80)
    ax1 = figure1.add_subplot(111)
    ax1.boxplot(df[name_col])
    boxplot1 = FigureCanvasTkAgg(figure1, visualScreen)
    boxplot1.get_tk_widget().place(x=30, y=100)
    ax1.set_xlabel(name_column[index_col])
    ax1.set_title('BOXPLOT')
    # HISPLOT của thuộc tính được chọn
    figure2 = plt.Figure(figsize=(5, 4), dpi=80)
    ax2 = figure2.add_subplot(111)
    sns.histplot(df[name_col], kde=True, ax=ax2)
    hisplot2 = FigureCanvasTkAgg(figure2, visualScreen)
    hisplot2.get_tk_widget().place(x=465, y=100)
    ax2.set_xlabel(name_column[index_col])
    ax2.set_title('HISTPLOT')

    Button(visualScreen, text='Quay lại', command=visualScreen.destroy).place(x=20, y=660)

    visualScreen.mainloop()

def AnalyScreen():
    analyScreen = Tk()
    analyScreen.title('Phân loại bệnh Ung thư vú')
    analyScreen.geometry('600x700')

    Label(analyScreen, text='DỰ ĐOÁN BỆNH', font=('bold', 25)).place(x=170, y=20)

    # Load file để phân tích
    def UploadAction():
        filepath = filedialog.askopenfilename()
        info_data = pd.read_csv(filepath)
        global to_drop
        data = info_data.drop(to_drop, axis=1)
        data = scaler.transform(data)
        result = knn.predict(data)
        info_data['Result'] = result
        info_data.to_csv(filepath, index=False)
        tkinter.messagebox.showinfo(title="Kết quả", message=' Dự đoán hoàn thành ')

    def analy_data():
        # Lấy thông tin mới được nhập bởi người dùng
        for i in range(30):
            value = data_input[i].get("1.0","end-1c")
            if value != '':
                data_input[i] = value
            else:
                data_input[i] = '0'
        data = pd.DataFrame([data_input], columns=name_column).astype('float')
        global to_drop
        data = data.drop(to_drop, axis=1)
        data = scaler.transform(data)
        # Dự đoán kết quả
        result = knn.predict(data)
        if result == "['B']":
            tkinter.messagebox.showinfo(title="Kết quả", message=' Kết quả dự đoán:   LÀNH TÍNH   ( - ) ')
        else:
            tkinter.messagebox.showinfo(title="Kết quả", message=' Kết quả dự đoán:   ÁC TÍNH   ( + ) ')
    data_input = [None] * 30

    # Hiển thị ô nhập thông tin
    for i in range(15):
        Label(analyScreen, text=name_column[i], font=('bold', 12)).place(x=20, y=80 + 30 * i)
        Label(analyScreen, text=name_column[i + 15], font=('bold', 12)).place(x=320, y=80 + 30 * i)

        data_input[i] = Text(analyScreen,width=7,height=1)
        data_input[i].place(x=200, y=80+30*i)

        data_input[i+15] = Text(analyScreen, width=7, height=1)
        data_input[i+15].place(x=500, y=80 + 30 * i)

    Button(analyScreen, text='Phân tích', command=lambda: analy_data()).place(x=250, y=550)
    Button(analyScreen, text='Quay lại', command=analyScreen.destroy).place(x=20, y=660)

    Label(analyScreen, text='Dự đoán qua file dữ liệu: ',font=('bold',12)).place(x=150, y=600)
    Button(analyScreen,text='Chọn file',command=UploadAction).place(x=350,y=600)

    analyScreen.mainloop()

def HomeScreen():
    homeScreen = Tk()
    homeScreen.title('Ứng dụng phân loại bệnh ung thư vú')
    homeScreen.geometry('600x400')

    Label(homeScreen, text='CHỨC NĂNG', font=('bold', 25)).place(x=190, y=50)

    Label(homeScreen, text='Phân tích dữ liệu', font=('bold', 20)).place(x=50, y=150)
    Button(homeScreen, text='Chọn',command = lambda: VisualScreen(choosen.current()), width=10).place(x=115, y=240)

    Label(homeScreen, text='Dự đoán bệnh', font=('bold', 20)).place(x=350, y=150)
    Button(homeScreen, text='Chọn',command =lambda: AnalyScreen(), width=10).place(x=405, y=200)
    # Combobox chọn thuộc tính để phân tích
    choosen = ttk.Combobox(homeScreen, width=27)

    choosen['values'] = tuple(name_column)

    choosen.place(x=58, y=200)

    choosen.current(0)

    homeScreen.mainloop()

ML(df)
HomeScreen()