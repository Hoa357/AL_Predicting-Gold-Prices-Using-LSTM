import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model
from tkinter import messagebox, ttk, Label, Entry, Tk, Button
import os
import joblib
import glob
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================
# 1. Tìm file mô hình mới nhất
# ============================
def find_latest_model(file_pattern):
    files = glob.glob(file_pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# ============================
# 2. Load số ngày từ config.json
# ============================
def load_look_back():
    try:
        with open("config.json", "r") as json_file:
            config = json.load(json_file)
            return config.get("look_back", 60)  # Giá trị mặc định là 60 nếu không tìm thấy
    except Exception as e:
        print(f"Lỗi khi đọc config.json: {e}")
        return 60  # Mặc định nếu xảy ra lỗi

look_back = load_look_back()  # Đọc số ngày từ config.json

# ============================
# 3. Load mô hình mới nhất
# ============================
latest_lr_model_path = find_latest_model("linear_regression_*_days.pkl")
lr_model = joblib.load(latest_lr_model_path) if latest_lr_model_path else None

latest_lstm_model_path = find_latest_model("lstm_model_*_days.h5")
lstm_model = load_model(latest_lstm_model_path) if latest_lstm_model_path else None

# ============================
# 4. Tải dữ liệu và chuẩn hóa
# ============================
data = pd.read_csv('Goal_PriceCleaned.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Price']])

# ============================
# 5. Hàm dự đoán nhiều ngày cho LSTM
# ============================
def predict_lstm():
    try:
        input_data = scaled_data[-look_back:].copy()
        predictions = []

        for _ in range(look_back):
            X_input = np.reshape(input_data[-look_back:], (1, look_back, 1))
            predicted = lstm_model.predict(X_input)
            predictions.append(predicted[0, 0])
            input_data = np.append(input_data, predicted, axis=0)

        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Xóa bảng cũ
        for row in table.get_children():
            table.delete(row)

        # Thêm dữ liệu vào bảng
        for i, price in enumerate(predicted_prices):
            table.insert("", "end", values=(f"LSTM Ngày {i + 1}", f"{price[0]:.2f}"))

        # Hiển thị kết quả LSTM
        lbl_result_lstm.configure(text=f"Dự đoán {look_back} ngày (LSTM):")
        
        # Đánh giá mô hình LSTM
        evaluate_model_lstm(predicted_prices)

    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi dự đoán LSTM: {str(e)}")

# ============================
# 6. Hàm dự đoán Linear Regression
# ============================
def predict_lr():
    try:
        # Đầu vào ban đầu cho Linear Regression
        input_data = np.array([[float(Textbox_open.get()), float(Textbox_high.get()),
                                float(Textbox_low.get()), float(Textbox_volume.get()), 
                                float(Textbox_change.get())]])

        # Danh sách chứa các giá trị dự đoán
        predictions = []

        # Dự đoán nhiều ngày
        for _ in range(look_back):
            prediction = lr_model.predict(input_data)
            predictions.append(prediction[0])

            # Cập nhật lại input_data với giá trị mới
            input_data = np.array([[*input_data[0, 1:], float(prediction[0])]])

        # Xóa bảng cũ
        for row in table.get_children():
            table.delete(row)

        # Thêm kết quả vào bảng
        for i, price in enumerate(predictions):
            table.insert("", "end", values=(f"Linear Regression Ngày {i + 1}", f"{price:.2f}"))

        # Hiển thị kết quả Linear Regression
        lbl_result_lr.configure(text=f"Dự đoán giá (Linear Regression) cho {look_back} ngày:")
        
        # Đánh giá mô hình Linear Regression
        evaluate_model_lr(predictions)

    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi dự đoán Linear Regression: {str(e)}")

# ============================
# 7. Hàm đánh giá mô hình LSTM
# ============================
def evaluate_model_lstm(predicted_prices):
    actual_prices = data['Price'].values[-look_back:]  # Giá thực tế
    mae = mean_absolute_error(actual_prices, predicted_prices)
    
    
    messagebox.showinfo("Kết quả đánh giá LSTM", 
                        f"MAE: {mae:.2f}\n")

# ============================
# 8. Hàm đánh giá mô hình Linear Regression
# ============================
def evaluate_model_lr(predictions):
    # Giả sử bạn có giá thực tế cho Linear Regression
    actual_prices = data['Price'].values[-look_back:]  # Giá thực tế
    mae = mean_absolute_error(actual_prices, predictions)
   
    
    messagebox.showinfo("Kết quả đánh giá Linear Regression", 
                        f"MAE: {mae:.2f}\n")

# ============================
# 9. Giao diện Tkinter
# ============================
form = Tk()
form.title("Dự đoán giá vàng")
form.geometry("700x600")
form.configure(bg="#f4f4f4")

# Tiêu đề
header_label = Label(form, text="Dự đoán giá vàng", font=("Arial", 18, "bold"), bg="#f4f4f4", fg="#007acc")
header_label.pack(pady=10)

# Nhập liệu cho Linear Regression
frame_input = ttk.Frame(form)
frame_input.pack(pady=10)  # Tăng khoảng cách giữa các thành phần
Label(frame_input, text="Giá Mở:").grid(row=0, column=0, padx=10, pady=5)  # Tăng khoảng cách
Textbox_open = ttk.Entry(frame_input, width=15)
Textbox_open.grid(row=0, column=1, padx=10, pady=5)  # Tăng khoảng cách

Label(frame_input, text="Giá Cao:").grid(row=1, column=0, padx=10, pady=5)
Textbox_high = ttk.Entry(frame_input, width=15)
Textbox_high.grid(row=1, column=1, padx=10, pady=5)

Label(frame_input, text="Giá Thấp:").grid(row=2, column=0, padx=10, pady=5)
Textbox_low = ttk.Entry(frame_input, width=15)
Textbox_low.grid(row=2, column=1, padx=10, pady=5)

Label(frame_input, text="Khối Lượng:").grid(row=3, column=0, padx=10, pady=5)
Textbox_volume = ttk.Entry(frame_input, width=15)
Textbox_volume.grid(row=3, column=1, padx=10, pady=5)

Label(frame_input, text="Thay đổi %:").grid(row=4, column=0, padx=10, pady=5)
Textbox_change = ttk.Entry(frame_input, width=15)
Textbox_change.grid(row=4, column=1, padx=10, pady=5)

# Kết quả dự đoán
lbl_result_lr = Label(form, text="Dự đoán giá (Linear Regression): N/A", fg="blue", bg="#f4f4f4")
lbl_result_lr.pack(pady=5)

lbl_result_lstm = Label(form, text=f"Dự đoán {look_back} ngày (LSTM):", fg="green", bg="#f4f4f4")
lbl_result_lstm.pack(pady=5)

# Table hiển thị kết quả
columns = ("Ngày", "Giá Dự Đoán")
table = ttk.Treeview(form, columns=columns, show="headings", height=15)
table.heading("Ngày", text="Ngày Dự Đoán (X)")
table.heading("Giá Dự Đoán", text="Giá Dự Đoán (Y)")
table.pack(pady=10)

# Nút dự đoán
button_frame = ttk.Frame(form)
button_frame.pack(pady=10)
ttk.Button(button_frame, text="Dự đoán (Linear Regression)", command=predict_lr).grid(row=0, column=0, padx=10, pady=5)  # Tăng khoảng cách
ttk.Button(button_frame, text="Dự đoán (LSTM)", command=predict_lstm).grid(row=0, column=1, padx=10, pady=5)  # Tăng khoảng cách

# Chạy giao diện
form.mainloop()