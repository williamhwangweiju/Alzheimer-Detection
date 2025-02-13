from config import *


def baseline_model(data, result):
    start_time = time.time()
    print("\n🚀 Starting Baseline Model Training...")
    
    # Convert one-hot encoding to categorical labels
    y_labels = np.argmax(result, axis=1)  # Convert from one-hot to categorical labels

    # Train-test split with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        data, y_labels, test_size=0.15, shuffle=True, stratify=y_labels, random_state=60
    )

    # Reshape data into 2D feature vectors
    X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten (400, 128, 128, 3) -> (400, 49152)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Initialize Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    
    end_time = time.time()
    print(f"\n✅ Baseline Model Training Completed in {end_time - start_time:.2f} seconds.")
    
if __name__ == "__main__":
    baseline_model()
