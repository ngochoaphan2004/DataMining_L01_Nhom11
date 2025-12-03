"""Data mining.ipynb
Import thư viện cần thiết
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
cols_to_use = ["Crm Cd Desc", "DATE OCC", "TIME OCC", "AREA NAME", "LAT", "LON", "Premis Desc"]

try:
  df = pd.read_csv('./Crime_Data_Selected_Cleaned.csv',
                     usecols=cols_to_use)
except FileNotFoundError:
  print("Lỗi: Không tìm thấy file CSV. Hãy đảm bảo file đã được tải về và nằm đúng thư mục.")
  df = pd.DataFrame(columns=cols_to_use)

# Lấy 500.000 row ngẫu nhiên
df = df.sample(n=500000, random_state=42)

df.dropna(subset=cols_to_use , inplace=True)
df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
df['DayOfWeek'] = df['DATE OCC'].dt.day_name()
df['Month'] = df['DATE OCC'].dt.month_name()

def map_crime_category(crm_desc):
  """
  Nhóm 114 loại tội phạm chi tiết thành 4 nhóm lớn.
  Thứ tự rất quan trọng (ví dụ: 'THEFT OF IDENTITY' là FRAUD, không phải PROPERTY)
  """
  crm_desc = str(crm_desc).upper()

  # === NHÓM 1: LỪA ĐẢO / GIẢ MẠO (FRAUD) ===
  fraud_keywords = [
      'IDENTITY', 'FRAUD', 'BUNCO', 'FORGERY', 'COUNTERFEIT',
      'EMBEZZLEMENT', 'DISHONEST EMPLOYEE', 'CREDIT CARD'
  ]
  if any(keyword in crm_desc for keyword in fraud_keywords):
    return 'FRAUD'

  # === NHÓM 2: BẠO LỰC (VIOLENT) ===
  violent_keywords = [
      'ASSAULT', 'BATTERY', 'ROBBERY', 'HOMICIDE', 'KIDNAPPING',
      'INTIMATE PARTNER', 'THREATS', 'WEAPON', 'BRANDISH', 'SEXUAL',
      'RAPE', 'SODOMY', 'ORAL COPULATION', 'LEWD', 'INCEST', 'STALKING',
      'MANSLAUGHTER', 'FALSE IMPRISONMENT', 'CRIMINAL HOMICIDE',
      'PIMPING', 'PANDERING', 'CHILD ABUSE', 'HUMAN TRAFFICKING'
  ]
  if any(keyword in crm_desc for keyword in violent_keywords):
    return 'VIOLENT'

  # === NHÓM 3: TÀI SẢN (PROPERTY) ===
  property_keywords = [
      'THEFT', 'BURGLARY', 'VANDALISM', 'STOLEN', 'SHOPLIFTING',
      'PICKPOCKET', 'PURSE SNATCHING', 'TILL TAP', 'ARSON', 'TRESPASSING'
  ]
  if any(keyword in crm_desc for keyword in property_keywords):
    return 'PROPERTY'

  # === NHÓM 4: CÁC TỘI PHẠM KHÁC ===
  return 'OTHER_CRIME'

df['Crm Cd Desc'] = df['Crm Cd Desc'].apply(map_crime_category)

print("\n--- GOM NHÓM HOÀN TẤT ---")
print("Kết quả số lượng các nhóm mới:")

print(df['Crm Cd Desc'].value_counts())

print(f"Số lượng record: {len(df)}")
df.head(500000)

numeric_features = ['TIME OCC', 'LAT', 'LON']
categorical_features = ['AREA NAME', 'Premis Desc', 'DayOfWeek', 'Month']

features = numeric_features + categorical_features
target = 'Crm Cd Desc'

X = df[features]
y = df[target]

class_counts = df[target].value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()
if rare_classes:
  df = df[~df[target].isin(rare_classes)]
print(df[target].value_counts())

class_names = list(y.unique())
class_names.sort()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_transformer = 'passthrough'

categorical_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

param_grid = {
    'model__max_depth': [8, 10, 12, 15], 
    'model__n_estimators': [50, 100, 150],
    'model__min_samples_leaf': [5, 10, 20]
}

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])
max_depth_values = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, None]

train_scores = []
test_scores = []

print("=== Thử các giá trị max_depth của RandomForest ===")

for depth in max_depth_values:
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=depth,
        random_state=42,
        n_jobs=-1
    )

    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', rf_model)
    ])

    # Train
    pipeline_rf.fit(X_train, y_train)

    # Accuracy train
    y_pred_train = pipeline_rf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)

    # Accuracy test
    y_pred_test = pipeline_rf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    train_scores.append(train_acc)
    test_scores.append(test_acc)

    print(f"max_depth={depth}: Train={train_acc:.4f}, Test={test_acc:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(max_depth_values, train_scores, marker='o', label="Train Accuracy")
plt.plot(max_depth_values, test_scores, marker='o', label="Test Accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Ảnh hưởng của max_depth tới độ chính xác Random Forest")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('random_forest_find_max_depth.png')
print("Đã lưu biểu đồ xác định max_depth vào 'random_forest_find_max_depth.png'")

print("=== Tìm kiếm tham số tốt nhất (dùng K-Fold với k = 3) ===")

best_score = 0
best_params = None
results = []

for max_depth in param_grid['model__max_depth']:
    for n_est in param_grid['model__n_estimators']:
        for min_leaf in param_grid['model__min_samples_leaf']: # <--- Vòng lặp mới
            
            # Tạo Pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', RandomForestClassifier(
                    max_depth=max_depth, 
                    n_estimators=n_est, 
                    min_samples_leaf=min_leaf, # <--- Nhớ truyền biến này vào
                    class_weight='balanced', 
                    random_state=42, 
                    n_jobs=-1
                ))
            ])

            # Chấm điểm bằng Cross Validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='accuracy')
            score = cv_scores.mean()

            # Lưu kết quả đầy đủ
            results.append({
                'max_depth': max_depth, 
                'n_estimators': n_est, 
                'min_samples_leaf': min_leaf,
                'score': score
            })

            print(f"Depth: {max_depth}, Trees: {n_est}, Leaf: {min_leaf} -> CV Score: {score:.4f}")

            # Cập nhật Best Params
            if score > best_score:
                best_score = score
                best_params = {
                    'model__max_depth': max_depth, 
                    'model__n_estimators': n_est,
                    'model__min_samples_leaf': min_leaf
                }

print("\nBest parameters found: ", best_params)
print("Best CV Score: ", best_score)

if best_params is None:
    print("Warning: No best parameters found. Using default values.")
    best_params = {'model__max_depth': 10, 'model__n_estimators': 100, 'model__min_samples_leaf': 5}

best_leaf = best_params['model__min_samples_leaf']
print(f"Đang vẽ biểu đồ với cố định min_samples_leaf = {best_leaf} (Giá trị tối ưu nhất)...")

# Lọc dữ liệu chỉ lấy các kết quả có min_samples_leaf bằng với best_leaf
filtered_results = [r for r in results if r['min_samples_leaf'] == best_leaf]

max_depths = param_grid['model__max_depth']
n_estimators_list = param_grid['model__n_estimators']
min_samples_leaf_list = param_grid['model__min_samples_leaf']

plt.figure(figsize=(12, 8))

# Định nghĩa kiểu nét vẽ để phân biệt min_samples_leaf
line_styles = ['-', '--', ':'] # Liền, Đứt nét, Chấm
colors = ['blue', 'green', 'red'] # Màu cho n_estimators

# Vòng lặp để vẽ 9 đường
for i, n_est in enumerate(n_estimators_list):
    for j, min_leaf in enumerate(min_samples_leaf_list):
        
        # Lọc dữ liệu cho cặp (n_est, min_leaf) cụ thể
        scores = [r['score'] for r in results 
                  if r['n_estimators'] == n_est and r['min_samples_leaf'] == min_leaf]
        
        # Vẽ đường
        if len(scores) == len(max_depths):
            plt.plot(max_depths, scores, 
                     marker='o',
                     linestyle=line_styles[j % len(line_styles)],
                     color=colors[i % len(colors)],
                     label=f'Trees={n_est}, Leaf={min_leaf}')

plt.xlabel('max_depth')
plt.ylabel('CV Accuracy')
plt.title('Toàn cảnh hiệu năng Random Forest (9 Cấu hình)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Đưa chú thích ra ngoài cho đỡ che hình
plt.grid(True)
plt.tight_layout()
plt.savefig('random_forest_sweet_spot.png')
print("Đã lưu biểu đồ 9 đường vào 'random_forest_sweet_spot.png'")

# Tạo best_pipeline với best_params
best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        max_depth=best_params['model__max_depth'],
        n_estimators=best_params['model__n_estimators'],
        min_samples_leaf=best_params['model__min_samples_leaf'],
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])
best_pipeline.fit(X_train, y_train)

print("\n[Mô hình cuối cùng] Đánh giá với K-Fold k=10 trên tập TRAIN")
y_pred_cv = cross_val_predict(best_pipeline, X_train, y_train, cv=10)
print(classification_report(y_train, y_pred_cv, zero_division=0))

cv_accuracy = accuracy_score(y_train, y_pred_cv)
print(f"!!! Độ chính xác với 10-Fold CV trên TRAIN: {cv_accuracy * 100:.2f}%")

print("\n--- Kết quả trên tập TEST ---")
y_pred_test = best_pipeline.predict(X_test)
print(classification_report(y_test, y_pred_test, zero_division=0))

y_pred_train = best_pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"!!! Độ chính xác trên tập TRAIN: {train_accuracy * 100:.2f}%")

input_data = {
    'TIME OCC': [2300],
    'LAT': [34.044],
    'LON': [-118.24],
    'AREA NAME': ['Central'],
    'Premis Desc': ['STREET'],
    'DayOfWeek': ['Friday'],
    'Month': ['December']
}


new_case = pd.DataFrame(input_data)

print("-" * 40)
print("Đang phân tích dữ liệu đầu vào...")
print(new_case)
print("-" * 40)

prediction = best_pipeline.predict(new_case)
probability = best_pipeline.predict_proba(new_case)

result = prediction[0]

print(f"KẾT QUẢ DỰ ĐOÁN: {result}")

print("\nĐộ tin cậy (Xác suất):")
classes = best_pipeline.named_steps['model'].classes_ 
probs = probability[0]

for crime_type, prob in zip(classes, probs):
    print(f" - {crime_type}: {prob * 100:.2f}%")