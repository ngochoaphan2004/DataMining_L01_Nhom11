import pandas as pd
import numpy as np

# ======= 1. Đọc dữ liệu =======
file_path = r"C:\Users\Admin\Downloads\Crime_Data_from_2020_to_Present.csv"
data = pd.read_csv(file_path)

# ======= 2. Chọn các cột quan trọng =======
important_cols = ["Crm Cd Desc", "DATE OCC", "TIME OCC", "AREA NAME", "LAT", "LON", "Premis Desc"]
data = data[important_cols]

# ======= 3. Loại bỏ các dòng có giá trị null =======
data_cleaned = data.dropna(subset=important_cols)

# ======= 4. Chuyển đổi kiểu dữ liệu =======
data_cleaned["DATE OCC"] = pd.to_datetime(data_cleaned["DATE OCC"], errors='coerce')
data_cleaned = data_cleaned.dropna(subset=["DATE OCC"])

# ======= 5. Xử lý và phát hiện outlier của TIME OCC (chỉ TIME OCC) =======
# Quy đổi TIME OCC về dạng số 4 chữ số: HHMM
data_cleaned["TIME OCC"] = data_cleaned["TIME OCC"].astype(int)

# TIME OCC hợp lệ phải từ 0000 đến 2359 và phút < 60
# Tách giờ và phút
data_cleaned["HOUR"] = data_cleaned["TIME OCC"] // 100
data_cleaned["MINUTE"] = data_cleaned["TIME OCC"] % 100

# Điều kiện hợp lệ
valid_time = (data_cleaned["HOUR"] >= 0) & (data_cleaned["HOUR"] <= 23) & \
             (data_cleaned["MINUTE"] >= 0) & (data_cleaned["MINUTE"] <= 59)

# Outliers = TIME OCC không hợp lệ
time_occ_outliers = data_cleaned[~valid_time]
outlier_count = len(time_occ_outliers)

# Loại bỏ outlier
data_cleaned = data_cleaned[valid_time]

# Xóa cột HOUR và MINUTE sau khi xử lý
data_cleaned = data_cleaned.drop(columns=["HOUR", "MINUTE"])

# ======= 6. Tính toán thống kê mô tả =======
numeric_cols = ["LAT", "LON", "TIME OCC"]

summary_stats = data_cleaned[numeric_cols].describe().T

extra_stats = pd.DataFrame({
    "median": data_cleaned[numeric_cols].median(),
    "mode": data_cleaned[numeric_cols].mode().iloc[0],
    "midrange": (data_cleaned[numeric_cols].min() + data_cleaned[numeric_cols].max()) / 2
})

# ======= 7. Xuất file =======
output_path = r"C:\Users\Admin\Downloads\Crime_Data_Selected_Cleaned.csv"
data_cleaned.to_csv(output_path, index=False)

# ======= 8. In kết quả =======
print("\n=== SUMMARY STATISTICS ===")
print(summary_stats)

print("\n=== EXTRA STATISTICS (Median, Mode, Midrange) ===")
print(extra_stats)

print("\n=== INVALID TIME OCC OUTLIERS REMOVED ===")
print(f"TIME OCC outliers removed: {outlier_count}")
