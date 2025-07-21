import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Hàm đọc và xem thông tin cơ bản về dữ liệu
def load_and_explore_data(file_path):
    print(f"Đọc file: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Kích thước dữ liệu: {df.shape}")
    print("\nThông tin cơ bản:")
    print(df.info())
    print("\nThống kê mô tả:")
    print(df.describe())
    print("\nKiểm tra giá trị thiếu:")
    print(df.isnull().sum())
    return df

# Hàm xử lý dữ liệu từ file heart.csv
def preprocess_heart_data(df):
    # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
    data = df.copy()
    
    # Kiểm tra và xử lý giá trị thiếu
    if data.isnull().sum().sum() > 0:
        print("Đang xử lý giá trị thiếu...")
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        numeric_imputer = SimpleImputer(strategy='median')
        data[numeric_features] = numeric_imputer.fit_transform(data[numeric_features])
    
    # Mã hóa các biến phân loại (nếu có)
    # Trong heart.csv, hầu hết các biến đã được mã hóa dưới dạng số
    
    # Chuẩn hóa các đặc trưng số
    print("Đang chuẩn hóa các đặc trưng số...")
    features = data.drop('target', axis=1)
    target = data['target']
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Tạo pipeline xử lý dữ liệu
    preprocessor = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Áp dụng tiền xử lý
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

# Hàm xử lý dữ liệu từ file heart_cleveland_upload.csv
def preprocess_cleveland_data(df):
    # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
    data = df.copy()
    
    # Đổi tên cột condition thành target cho thống nhất
    if 'condition' in data.columns:
        data.rename(columns={'condition': 'target'}, inplace=True)
    
    # Kiểm tra và xử lý giá trị thiếu
    if data.isnull().sum().sum() > 0:
        print("Đang xử lý giá trị thiếu...")
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        numeric_imputer = SimpleImputer(strategy='median')
        data[numeric_features] = numeric_imputer.fit_transform(data[numeric_features])
    
    # Chuẩn hóa các đặc trưng số
    print("Đang chuẩn hóa các đặc trưng số...")
    features = data.drop('target', axis=1)
    target = data['target']
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Tạo pipeline xử lý dữ liệu
    preprocessor = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Áp dụng tiền xử lý
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

# Hàm xử lý dữ liệu từ file heart_disease_uci.csv
def preprocess_uci_data(df):
    # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
    data = df.copy()
    
    # Xử lý giá trị thiếu
    if data.isnull().sum().sum() > 0:
        print("Đang xử lý giá trị thiếu...")
        # Xử lý các cột số
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        numeric_imputer = SimpleImputer(strategy='median')
        data[numeric_features] = numeric_imputer.fit_transform(data[numeric_features])
        
        # Xử lý các cột phân loại
        categorical_features = data.select_dtypes(include=['object']).columns
        if len(categorical_features) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            data[categorical_features] = categorical_imputer.fit_transform(data[categorical_features])
    
    # Mã hóa các biến phân loại
    print("Đang mã hóa các biến phân loại...")
    categorical_features = ['sex', 'cp', 'restecg', 'slope', 'thal', 'dataset']
    categorical_features = [col for col in categorical_features if col in data.columns]
    
    if len(categorical_features) > 0:
        # Sử dụng OneHotEncoder để mã hóa các biến phân loại
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(data[categorical_features])
        encoded_df = pd.DataFrame(
            encoded_data, 
            columns=encoder.get_feature_names_out(categorical_features)
        )
        
        # Loại bỏ các cột phân loại gốc và thêm các cột đã mã hóa
        data = data.drop(categorical_features, axis=1)
        data = pd.concat([data, encoded_df], axis=1)
    
    # Xử lý cột target (num)
    if 'num' in data.columns:
        # Chuyển thành dạng nhị phân: 0 = không bệnh (0), 1 = có bệnh (1-4)
        data['target'] = data['num'].apply(lambda x: 0 if x == 0 else 1)
        data = data.drop('num', axis=1)
    
    # Loại bỏ các cột không cần thiết
    if 'id' in data.columns:
        data = data.drop('id', axis=1)
    
    # Chuẩn hóa các đặc trưng số
    print("Đang chuẩn hóa các đặc trưng số...")
    features = data.drop(['target'], axis=1, errors='ignore')
    target = data['target']
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Tạo pipeline xử lý dữ liệu
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Tạo preprocessor chỉ cho các cột số
    preprocessor = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    if len(numeric_features) > 0:
        X_train_processed = preprocessor.fit_transform(X_train[numeric_features])
        X_test_processed = preprocessor.transform(X_test[numeric_features])
    else:
        X_train_processed = X_train.values
        X_test_processed = X_test.values
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

# Hàm trực quan hóa dữ liệu
def visualize_data(df, file_name):
    # Tạo thư mục để lưu hình ảnh
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Phân bố các đặc trưng số
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    for feature in numeric_features:
        plt.figure(figsize=(10, 6))
        plt.hist(df[feature], bins=30)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.savefig(f'visualizations/{file_name}_{feature}_hist.png')
        plt.close()
    
    # 2. Bản đồ tương quan
    plt.figure(figsize=(12, 10))
    corr = df[numeric_features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(f'visualizations/{file_name}_correlation.png')
    plt.close()
    
    # 3. Phân tích mục tiêu (nếu có)
    if 'target' in df.columns or 'condition' in df.columns:
        target_col = 'target' if 'target' in df.columns else 'condition'
        plt.figure(figsize=(8, 6))
        target_counts = df[target_col].value_counts()
        target_counts.plot(kind='bar')
        plt.title('Target Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig(f'visualizations/{file_name}_target_dist.png')
        plt.close()
        
        # 4. Boxplot cho các đặc trưng số theo mục tiêu
        for feature in numeric_features[:5]:  # Chỉ lấy 5 đặc trưng đầu tiên để tránh quá nhiều đồ thị
            if feature != target_col:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=target_col, y=feature, data=df)
                plt.title(f'Boxplot of {feature} by Target')
                plt.savefig(f'visualizations/{file_name}_{feature}_boxplot.png')
                plt.close()

# Hàm kết hợp dữ liệu từ nhiều nguồn
def combine_datasets(datasets):
    # Kiểm tra xem có thể kết hợp các tập dữ liệu không
    common_columns = set.intersection(*[set(df.columns) for df in datasets])
    if len(common_columns) == 0:
        print("Không có cột chung giữa các tập dữ liệu!")
        return None
    
    # Chỉ giữ các cột chung
    combined_data = pd.concat([df[list(common_columns)] for df in datasets], ignore_index=True)
    print(f"Đã kết hợp {len(datasets)} tập dữ liệu với {combined_data.shape[0]} mẫu và {len(common_columns)} đặc trưng.")
    
    return combined_data

# Hàm chính để xử lý tất cả các tập dữ liệu
def main():
    # 1. Đọc và khám phá dữ liệu
    heart_df = load_and_explore_data('heart.csv')
    cleveland_df = load_and_explore_data('heart_cleveland_upload.csv')
    uci_df = load_and_explore_data('heart_disease_uci.csv')
    
    # 2. Trực quan hóa dữ liệu
    visualize_data(heart_df, 'heart')
    visualize_data(cleveland_df, 'cleveland')
    visualize_data(uci_df, 'uci')
    
    # 3. Xử lý từng tập dữ liệu
    print("\n--- Xử lý dữ liệu từ heart.csv ---")
    heart_X_train, heart_X_test, heart_y_train, heart_y_test, heart_preprocessor = preprocess_heart_data(heart_df)
    
    print("\n--- Xử lý dữ liệu từ heart_cleveland_upload.csv ---")
    cleveland_X_train, cleveland_X_test, cleveland_y_train, cleveland_y_test, cleveland_preprocessor = preprocess_cleveland_data(cleveland_df)
    
    print("\n--- Xử lý dữ liệu từ heart_disease_uci.csv ---")
    uci_X_train, uci_X_test, uci_y_train, uci_y_test, uci_preprocessor = preprocess_uci_data(uci_df)
    
    # 4. Lưu dữ liệu đã xử lý
    np.save('processed_data/heart_X_train.npy', heart_X_train)
    np.save('processed_data/heart_X_test.npy', heart_X_test)
    np.save('processed_data/heart_y_train.npy', heart_y_train)
    np.save('processed_data/heart_y_test.npy', heart_y_test)
    
    np.save('processed_data/cleveland_X_train.npy', cleveland_X_train)
    np.save('processed_data/cleveland_X_test.npy', cleveland_X_test)
    np.save('processed_data/cleveland_y_train.npy', cleveland_y_train)
    np.save('processed_data/cleveland_y_test.npy', cleveland_y_test)
    
    np.save('processed_data/uci_X_train.npy', uci_X_train)
    np.save('processed_data/uci_X_test.npy', uci_X_test)
    np.save('processed_data/uci_y_train.npy', uci_y_train)
    np.save('processed_data/uci_y_test.npy', uci_y_test)
    
    # 5. Thử kết hợp các tập dữ liệu (nếu có thể)
    print("\n--- Thử kết hợp các tập dữ liệu ---")
    # Chuẩn hóa các tên cột trước khi kết hợp
    heart_df_normalized = heart_df.copy()
    cleveland_df_normalized = cleveland_df.copy()
    
    # Đổi tên cột condition thành target trong cleveland_df (nếu có)
    if 'condition' in cleveland_df_normalized.columns:
        cleveland_df_normalized.rename(columns={'condition': 'target'}, inplace=True)
    
    # Thử kết hợp heart_df và cleveland_df (vì chúng có cấu trúc tương tự)
    combined_df = combine_datasets([heart_df_normalized, cleveland_df_normalized])
    
    if combined_df is not None:
        print("\n--- Xử lý dữ liệu kết hợp ---")
        # Trực quan hóa dữ liệu kết hợp
        visualize_data(combined_df, 'combined')
        
        # Xử lý dữ liệu kết hợp (giả sử cấu trúc giống heart.csv)
        combined_X_train, combined_X_test, combined_y_train, combined_y_test, combined_preprocessor = preprocess_heart_data(combined_df)
        
        # Lưu dữ liệu kết hợp đã xử lý
        np.save('processed_data/combined_X_train.npy', combined_X_train)
        np.save('processed_data/combined_X_test.npy', combined_X_test)
        np.save('processed_data/combined_y_train.npy', combined_y_train)
        np.save('processed_data/combined_y_test.npy', combined_y_test)
    
    print("\nQuá trình xử lý dữ liệu hoàn tất!")

# Tạo thư mục lưu dữ liệu đã xử lý
import os
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')

# Chạy hàm chính
if __name__ == "__main__":
    main() 