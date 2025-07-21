# Phân tích Dữ liệu Bệnh Tim với Học Máy

Dự án này thực hiện việc xử lý, phân tích và xây dựng mô hình học máy trên các bộ dữ liệu về bệnh tim để dự đoán khả năng mắc bệnh tim.

## Cấu trúc dự án

```
.
├── data_preprocessing.py    # Script tiền xử lý dữ liệu
├── model_training.py        # Script huấn luyện và đánh giá mô hình
├── model_explanation.py     # Script giải thích mô hình
├── create_name_mapping.py   # Script tạo file ánh xạ tên đặc trưng
├── heart.csv                # Dữ liệu bệnh tim cơ bản
├── heart_cleveland_upload.csv # Dữ liệu từ Cleveland Clinic
├── heart_disease_uci.csv    # Dữ liệu từ UCI repository
├── processed_data/          # Thư mục chứa dữ liệu đã xử lý
├── model_results/           # Thư mục chứa kết quả mô hình
├── model_explanations/      # Thư mục chứa giải thích mô hình
└── visualizations/          # Thư mục chứa hình ảnh trực quan hóa
```

## Bộ dữ liệu

Dự án sử dụng ba bộ dữ liệu về bệnh tim:

1. **heart.csv**: Bộ dữ liệu cơ bản về bệnh tim với 13 đặc trưng, 303 mẫu
2. **heart_cleveland_upload.csv**: Dữ liệu từ Cleveland Clinic, 297 mẫu
3. **heart_disease_uci.csv**: Dữ liệu từ UCI Machine Learning Repository, 920 mẫu

## Chi tiết quy trình xử lý dữ liệu

### 1. Khám phá và trực quan hóa dữ liệu

Script `data_preprocessing.py` bắt đầu bằng việc đọc và hiển thị thông tin cơ bản về mỗi tập dữ liệu:
- Kích thước (số hàng và cột)
- Kiểu dữ liệu và thông tin cơ bản
- Thống kê mô tả (trung bình, độ lệch chuẩn, min/max)
- Số lượng giá trị thiếu trong mỗi cột

Tiếp theo, script tạo các biểu đồ trực quan:
- Phân bố cho từng đặc trưng số học
- Ma trận tương quan giữa các đặc trưng
- Phân bố mục tiêu (tỷ lệ có/không có bệnh tim)
- Boxplot cho mỗi đặc trưng được phân nhóm theo mục tiêu

### 2. Xử lý dữ liệu bị thiếu

Dữ liệu bị thiếu được xử lý khác nhau tùy vào loại dữ liệu:

**Đối với tất cả các tập dữ liệu:**
- **Dữ liệu số**: Sử dụng `SimpleImputer` với chiến lược `median` (giá trị trung vị)
  ```python
  numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
  numeric_imputer = SimpleImputer(strategy='median')
  data[numeric_features] = numeric_imputer.fit_transform(data[numeric_features])
  ```

**Đặc biệt với tập dữ liệu UCI:**
- **Dữ liệu phân loại**: Sử dụng `SimpleImputer` với chiến lược `most_frequent` (giá trị phổ biến nhất)
  ```python
  categorical_features = data.select_dtypes(include=['object']).columns
  categorical_imputer = SimpleImputer(strategy='most_frequent')
  data[categorical_features] = categorical_imputer.fit_transform(data[categorical_features])
  ```

Việc sử dụng giá trị trung vị thay vì trung bình giúp giảm ảnh hưởng của các giá trị ngoại lai, điều này đặc biệt quan trọng trong dữ liệu y tế.

### 3. Mã hóa biến phân loại

Đối với các biến phân loại (đặc biệt trong UCI dataset):
```python
categorical_features = ['sex', 'cp', 'restecg', 'slope', 'thal', 'dataset']
```

Script sử dụng `OneHotEncoder` để biến đổi các biến phân loại thành dạng số nhị phân, tạo ra các cột mới cho mỗi giá trị phân loại.

### 4. Chuẩn hóa đặc trưng

Tất cả các đặc trưng số được chuẩn hóa bằng `StandardScaler` để đưa chúng về cùng thang đo:
```python
preprocessor = Pipeline([
    ('scaler', StandardScaler())
])
```

### 5. Chia dữ liệu và lưu trữ

Mỗi tập dữ liệu được chia thành tập huấn luyện (80%) và tập kiểm tra (20%):
```python
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)
```

Dữ liệu đã xử lý được lưu dưới dạng file NumPy trong thư mục `processed_data/`.

### 6. Kết hợp dữ liệu

Script cũng thử kết hợp `heart.csv` và `heart_cleveland_upload.csv` dựa trên các cột chung để tạo một tập dữ liệu lớn hơn để huấn luyện.

## Chi tiết quy trình huấn luyện mô hình

Script `model_training.py` huấn luyện bốn loại mô hình khác nhau:

### 1. Logistic Regression

```python
# Các tham số được tối ưu hóa
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 500, 1000]
}
```

### 2. Random Forest

```python
# Các tham số được tối ưu hóa
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

### 3. Support Vector Machine (SVM)

```python
# Các tham số được tối ưu hóa
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'degree': [2, 3, 4]
}
```

### 4. Neural Network

```python
# Các tham số được tối ưu hóa
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}
```

### Tối ưu hóa siêu tham số

Tất cả các mô hình đều sử dụng `GridSearchCV` với 5-fold cross-validation để tìm ra bộ siêu tham số tối ưu:
```python
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
```

### Đánh giá mô hình

Mỗi mô hình được đánh giá dựa trên các độ đo:
- Accuracy (độ chính xác)
- Precision (độ chính xác dương tính)
- Recall (độ nhạy/thu hồi)
- F1 Score (điểm F1)
- ROC AUC (diện tích dưới đường cong ROC)
- Confusion Matrix (ma trận nhầm lẫn)

Các biểu đồ được tạo ra bao gồm:
- Ma trận nhầm lẫn
- Đường cong ROC
- Đường cong Precision-Recall

## Chi tiết quy trình giải thích mô hình

Script `model_explanation.py` giải thích các dự đoán của mô hình bằng hai phương pháp chính:

### 1. Phân tích tầm quan trọng của đặc trưng

**Logistic Regression:**
- Sử dụng giá trị tuyệt đối của các hệ số (coefficients) để xác định tầm quan trọng của đặc trưng

**Random Forest:**
- Sử dụng thuộc tính `feature_importances_` dựa trên độ giảm bất tinh khiết Gini

**Tất cả các mô hình:**
- Sử dụng `permutation_importance` để đo lường tầm quan trọng của đặc trưng bằng cách hoán vị giá trị của từng đặc trưng và quan sát tác động đến hiệu suất mô hình

### 2. SHAP values (SHapley Additive exPlanations)

SHAP values cung cấp cách giải thích từng dự đoán cụ thể:

- **Logistic Regression:** `LinearExplainer`
- **Random Forest:** `TreeExplainer`
- **SVM/Neural Network:** `KernelExplainer`

Các biểu đồ SHAP được tạo ra bao gồm:
- SHAP Summary Plot: Hiển thị tổng quan về tác động của đặc trưng
- SHAP Force Plot: Giải thích từng dự đoán cụ thể

### 3. Tổng hợp đặc trưng quan trọng

Script cũng tổng hợp các đặc trưng quan trọng nhất giữa các mô hình để cung cấp cái nhìn tổng quát về các yếu tố quyết định.

## Tệp ánh xạ đặc trưng (Feature Mapping)

Script `create_name_mapping.py` tạo ra file `name_mapping.csv` để cung cấp thông tin về các đặc trưng:
- Chỉ số đặc trưng
- Tên đặc trưng có ý nghĩa
- Mô tả chi tiết về ý nghĩa của đặc trưng

Tệp này được sử dụng trong quá trình giải thích mô hình để hiển thị tên đặc trưng có ý nghĩa trong các biểu đồ và báo cáo.

## Yêu cầu hệ thống

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- joblib >= 1.1.0
- shap >= 0.40.0

Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

## Quy trình thực hiện đầy đủ

### Bước 1: Tạo ánh xạ tên đặc trưng
```bash
python create_name_mapping.py
```
Output: `name_mapping.csv`

### Bước 2: Tiền xử lý dữ liệu
```bash
python data_preprocessing.py
```
Output:
- Thư mục `processed_data/` với các file dữ liệu đã xử lý
- Thư mục `visualizations/` với các biểu đồ trực quan hóa dữ liệu

### Bước 3: Huấn luyện và đánh giá mô hình
```bash
python model_training.py
```
Output:
- Thư mục `model_results/` với các mô hình đã huấn luyện (.pkl)
- Biểu đồ ma trận nhầm lẫn, ROC curve và precision-recall curve
- File so sánh hiệu suất các mô hình

### Bước 4: Giải thích mô hình
```bash
python model_explanation.py
```
Output:
- Thư mục `model_explanations/` với các biểu đồ tầm quan trọng đặc trưng và SHAP
- File CSV tổng hợp đặc trưng quan trọng
- HTML SHAP force plots

## Giải thích đầy đủ các đặc trưng trong dữ liệu bệnh tim

- **age**: Tuổi của bệnh nhân (năm)
- **sex**: Giới tính (1 = nam, 0 = nữ)
- **cp**: Loại đau ngực
  - 0: Đau ngực điển hình
  - 1: Đau ngực không điển hình
  - 2: Đau không liên quan đến đau thắt ngực
  - 3: Không có triệu chứng
- **trestbps**: Huyết áp lúc nghỉ (mm Hg)
- **chol**: Cholesterol huyết thanh (mg/dl)
- **fbs**: Đường huyết lúc đói > 120 mg/dl (1 = đúng, 0 = sai)
- **restecg**: Kết quả điện tâm đồ lúc nghỉ
  - 0: Bình thường
  - 1: Có bất thường sóng ST-T
  - 2: Hiển thị phì đại tâm thất trái
- **thalach**: Nhịp tim tối đa đạt được
- **exang**: Đau thắt ngực do tập thể dục (1 = có, 0 = không)
- **oldpeak**: Sự chênh lệch đoạn ST do tập thể dục
- **slope**: Độ dốc của đoạn ST tập thể dục
  - 0: Dốc lên
  - 1: Phẳng
  - 2: Dốc xuống
- **ca**: Số lượng mạch máu chính (0-3)
- **thal**: Thalassemia
  - 0: NULL
  - 1: Khiếm khuyết cố định
  - 2: Bình thường
  - 3: Khiếm khuyết có thể đảo ngược
- **target/condition**: Biến mục tiêu (1 = có bệnh tim, 0 = không có bệnh tim)

## Lưu ý khi huấn luyện

1. Quá trình tối ưu hóa siêu tham số có thể mất nhiều thời gian, đặc biệt là với SVM và Neural Network.
2. Với tập dữ liệu lớn (như combined dataset), việc tính toán SHAP values có thể tốn nhiều bộ nhớ. Trong trường hợp này, script đã được thiết kế để giới hạn số lượng mẫu sử dụng.
3. Kết quả của mô hình có thể thay đổi nhẹ giữa các lần chạy do yếu tố ngẫu nhiên trong quá trình chia dữ liệu và huấn luyện, mặc dù đã cố định random_state=42.
4. Kết quả tốt nhất thường đến từ tập dữ liệu kết hợp, nhưng điều này có thể khác nhau tùy thuộc vào tính chất cụ thể của dữ liệu. 

Tất cả các đặc trưng số được chuẩn hóa bằng StandardScaler để đưa chúng về cùng thang đo (trung bình = 0, độ lệch chuẩn = 1). Việc này rất quan trọng cho các thuật toán học máy như SVM hoặc mạng nơ-ron.