from data_preprocessing import load_data, preprocess_data
from model import train_and_evaluate

def main():
    """Main function to run the training pipeline."""
    data_path = "data/tested.csv"  # Update this path if needed
    df = load_data(data_path)
    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df)
    train_and_evaluate(preprocessor, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
