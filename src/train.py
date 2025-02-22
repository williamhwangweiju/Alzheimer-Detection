from config import *  
from utils import preprocess_data
from data_loader import load_data  

def train():
    print("🚀 Training process started...")

    # Load dataset
    data = load_data()  # Ensure this function returns data
    print(f"✅ Data Loaded: {type(data)}")

    # Preprocess dataset
    data, result= preprocess_data(data)
    # print(f"✅ Preprocessing Completed - X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")

    # TODO: Add model training logic here
    print("🔧 Model training logic goes here...\n")
        
if __name__ == "__main__":
    train()
