from config import *  # Ensure correct import path
import time

# folder_path = '/root/.cache/kagglehub/datasets/ninadaithal/imagesoasis/versions/1/Data'
dataset_per_cat = 400

def data_management(folder_path):
    print("🔍 Scanning dataset directories...")
    rows = 4  # Number of categories
    array = [[] for _ in range(rows)]  # Create a list of empty lists
    count = 0  # Keeps track of categories

    for folder in os.listdir(folder_path):
        if count >= rows:
            break  # Ensure we don't exceed the index limit

        print(f"📂 Processing category {count + 1}: {folder}")

        for dirname, _, filenames in os.walk(os.path.join(folder_path, folder)):
            for filename in filenames:
                array[count].append(os.path.join(dirname, filename))

        print(f"📸 Found {len(array[count])} images for category {count + 1}")

        count += 1  # Move to the next category

    # Limit dataset per category
    for index in range(rows):
        array[index] = array[index][:dataset_per_cat]
        print(f"✔ Limited to {dataset_per_cat} images for category {index + 1}")

    print("✅ Data management complete.\n")
    return array

def resize_and_encode(array):
    print("🖼️ Resizing images and encoding labels...")
    encoder = OneHotEncoder(sparse_output=False)  # Fix OneHotEncoder usage
    encoded_labels = encoder.fit_transform([[0], [1], [2], [3]])  # Pre-fit encoder

    data = []
    result = []

    for index in range(len(array)):  # Iterate correctly over categories
        print(f"📌 Processing category {index + 1}...")
        for path in array[index]:
            try:
                img = Image.open(path).convert("RGB")  # Ensure images are in RGB mode
                img = img.resize((128, 128))
                img = np.array(img)

                if img.shape == (128, 128, 3):  # Ensure valid image shape
                    data.append(img)
                    result.append(encoded_labels[index])  # Use pre-encoded labels

            except Exception as e:
                print(f"⚠️ Skipping {path} due to error: {e}")

    data = np.array(data)
    result = np.array(result)

    print(f"✅ Resizing and encoding complete: {len(data)} images processed.\n")
    return data, result

def split_data(data, result, test_size=0.15):
    print("📊 Splitting dataset into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=test_size, shuffle=True, random_state=60)
    print(f"✔ Training set: {X_train.shape[0]} samples")
    print(f"✔ Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

def preprocess_data(path):
    print("🚀 Starting full data processing pipeline...\n")

    start_time = time.time()

    array = data_management(path)
    data, result = resize_and_encode(array)
    # X_train, X_test, y_train, y_test = split_data(data, result, 0.25)

    end_time = time.time()
    print(f"\n✅ Data processing completed in {end_time - start_time:.2f} seconds")
    
    # print("\n📌 Final Dataset Summary:")
    # print(f"🔹 Training Data: {X_train.shape}, Labels: {y_train.shape}")
    # print(f"🔹 Test Data: {X_test.shape}, Labels: {y_test.shape}")

    # return X_train, X_test, y_train, y_test
    return data, result

# Correct syntax for executing script
if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = preprocess_data()
    data, result = preprocess_data()
