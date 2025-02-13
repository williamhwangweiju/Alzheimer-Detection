import kagglehub

def load_data():
    # Download dataset
    path = kagglehub.dataset_download("ninadaithal/imagesoasis")

    print("✅ Dataset downloaded to:", path)

    return path  # Return the correct path

# Run download
if __name__ == "__main__":
    dataset_path = load_data()
    print(f"📂 Check your dataset files inside: {dataset_path}")
