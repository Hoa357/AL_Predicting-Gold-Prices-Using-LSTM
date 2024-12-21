import pandas as pd

# Đọc dữ liệu từ file CSV
data = pd.read_csv('GiaVang_Full_2003_2024.csv')

# Hàm xử lý ngày tháng không đồng nhất
def correct_date_format(date_str):
    # Kiểm tra định dạng ngày tháng
    for fmt in ('%d/%m/%Y', '%m/%d/%Y'):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return pd.NaT  # Trả về NaT nếu không thể chuyển đổi

# Áp dụng hàm chuẩn hóa cho cột 'Date'
data['Date'] = data['Date'].apply(correct_date_format)

# Kiểm tra các giá trị NaT (ngày không hợp lệ sau khi chuyển đổi)
if data['Date'].isna().sum() > 0:
    print("Có lỗi trong việc chuyển đổi một số giá trị ngày tháng:")
    print(data[data['Date'].isna()])

# Đặt lại cột 'Date' làm chỉ mục
data.set_index('Date', inplace=True)

# Xử lý dữ liệu cột 'Vol.' và 'Change%' 
def process_vol(volume_str):
    if isinstance(volume_str, str):
        if 'K' in volume_str:
            return round(float(volume_str.replace('K', '').strip()) * 1000, 0)
        return round(float(volume_str), 0)
    return volume_str

def process_change(change_str):
    if isinstance(change_str, str):
        return round(float(change_str.replace('%', '').strip()) / 100, 4)
    elif isinstance(change_str, (int, float)):
        return round(float(change_str) / 100, 4)
    return None

# Xử lý cột 'Vol.' và 'Change%' bằng các hàm đã định nghĩa
data['Vol.'] = data['Vol.'].apply(process_vol)
data['Change%'] = data['Change%'].apply(process_change)

# Loại bỏ dấu phẩy và chuyển thành float cho các cột 'Price', 'Open', 'High', 'Low'
def clean_price(price_str):
    if isinstance(price_str, str):
        return round(float(price_str.replace(',', '')), 2)
    return price_str

# Áp dụng hàm làm sạch dữ liệu cho các cột số
price_columns = ['Price', 'Open', 'High', 'Low']
for col in price_columns:
    data[col] = data[col].apply(clean_price)

# Đổi lại định dạng cột 'Date' thành 'YYYY-MM-DD'
data.index = data.index.strftime('%Y-%m-%d')

# Lưu dữ liệu đã chuẩn hóa vào file CSV mới
data.to_csv('Goal_PriceCleaned.csv')

print("Dữ liệu đã được chuẩn hóa và lưu vào file 'Goal_PriceCleaned.csv'")