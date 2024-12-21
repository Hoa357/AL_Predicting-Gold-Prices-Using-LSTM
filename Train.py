import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tkinter import messagebox, ttk, Label, Entry, Tk
import tkinter as tk
import os
import joblib
import json  # Để lưu số ngày dự đoán

# ============================
# 1. Tải và chuẩn bị dữ liệu
# ============================
data = pd.read_csv('Goal_PriceCleaned.csv')

# Chuyển đổi cột 'Date' thành datetime và đặt làm index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Kiểm tra và xử lý giá trị NaN
if data.isnull().sum().sum() > 0:
    data = data.dropna()

# Định nghĩa các cột đặc trưng và mục tiêu
features = ['Open', 'High', 'Low', 'Vol.', 'Change%']
target = 'Price'

# Chuẩn hóa dữ liệu cho LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Price']])

# ============================
# 2. Hàm tạo tập dữ liệu LSTM
# ============================
def create_lstm_dataset(data, look_back):
    x, y = [], []
    for i in range(len(data) - look_back):
        x.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(x), np.array(y)

# ============================
# 3. Hàm huấn luyện mô hình
# ============================
def train_models():
    try:
        look_back = int(Textbox_days.get())
        if look_back <= 0:
            messagebox.showerror("Lỗi", "Số ngày phải là số nguyên dương!")
            return
        
        if len(data) < look_back:
            messagebox.showerror("Lỗi", "Không đủ dữ liệu cho số ngày đã nhập!")
            return
        
        # Lưu số ngày dự đoán vào file JSON
        config = {"look_back": look_back}
        with open("config.json", "w") as json_file:
            json.dump(config, json_file)
        
        # ----- Huấn luyện Linear Regression -----
        subset_data = data[-look_back:]
        X_lr = subset_data[features]
        y_lr = subset_data[target]

        # Chia dữ liệu train/test
        X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, shuffle=True)
        
        # Huấn luyện mô hình
        lr_model = LinearRegression()
        lr_model.fit(X_train_lr, y_train_lr)
        lr_filename = f'linear_regression_{look_back}_days.pkl'
        joblib.dump(lr_model, lr_filename)
        
        # ----- Huấn luyện LSTM -----
        x_lstm, y_lstm = create_lstm_dataset(scaled_data, look_back)
        x_lstm = np.reshape(x_lstm, (x_lstm.shape[0], x_lstm.shape[1], 1))
        
        # Chia dữ liệu
        train_size = int(len(x_lstm) * 0.8)
        x_train_lstm, x_test_lstm = x_lstm[:train_size], x_lstm[train_size:]
        y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

        # Tạo mô hình LSTM
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(x_train_lstm, y_train_lstm, batch_size=16, epochs=10, verbose=1)
        lstm_filename = f'lstm_model_{look_back}_days.h5'
        lstm_model.save(lstm_filename)
        
        # Thông báo thành công
        messagebox.showinfo("Thành công", f"Hai mô hình đã được lưu:\n- {lr_filename}\n- {lstm_filename}\n- config.json")
    
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng nhập số ngày hợp lệ!")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {str(e)}")

# ============================
# 4. Tạo giao diện Tkinter
# ============================
form = Tk()
form.title("Huấn luyện mô hình")
form.geometry("500x250")
form.configure(bg="#f4f4f4")

# Tiêu đề
header_label = Label(form, text="Huấn luyện mô hình LSTM & Linear Regression", font=("Arial", 14, "bold"), bg="#f4f4f4", fg="#007acc")
header_label.pack(pady=10)

# Nhập số ngày dự đoán
frame_input = ttk.Frame(form)
frame_input.pack(pady=20)
Label(frame_input, text="Số ngày dự đoán:", font=("Arial", 12)).grid(row=0, column=0, padx=10)
Textbox_days = ttk.Entry(frame_input, font=("Arial", 12), width=10)
Textbox_days.grid(row=0, column=1, padx=10)

# Nút huấn luyện
btn_train = ttk.Button(form, text="Huấn luyện mô hình", command=train_models)
btn_train.pack(pady=10)

# Chạy giao diện
form.mainloop()
