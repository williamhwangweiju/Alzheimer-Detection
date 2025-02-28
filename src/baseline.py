from config import *  
from utils import preprocess_data
from data_loader import load_data 

def baseline():
    print("🚀 Training process started...")

    # Load dataset
    path = load_data()
    print(f"✅ Data Loaded: {path}")

    # Preprocess dataset
    data, result = preprocess_data(path)

    # Convert one-hot encoding to categorical labels
    y_labels = np.argmax(result, axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        data, y_labels, test_size=0.15, shuffle=True, stratify=y_labels, random_state=60
    )
    
    # Reshape data into 2D feature vectors
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Initialize Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    # Generate and plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Non-Demented", "Mild Dementia", "Moderate Dementia", "Very Mild Dementia"],
                yticklabels=["Non-Demented", "Mild Dementia", "Moderate Dementia", "Very Mild Dementia"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

if __name__ == "__main__":
    baseline()