# 📊 Titanic Survival Prediction

This project uses a machine learning pipeline to predict passenger survival on the Titanic using a dataset with numerical and categorical features. It includes preprocessing, training using a Random Forest Classifier, and model evaluation.

## 📁 Project Structure

```
├── data_preprocessing.py   # Handles data loading and preprocessing
├── model.py                # Model training and evaluation
├── main.py                 # Main script to execute pipeline
├── tested.csv              # Titanic dataset (local or external)
├── models/
│   └── final_model.pkl     # Trained model saved after execution
```

## ⚙️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
pandas
scikit-learn
joblib
```

## 🚀 How to Run

1. Make sure your Titanic dataset is available and accessible. If using the `tested.csv`, ensure it’s correctly referenced in `main.py`.
2. Run the training pipeline:

```bash
python main.py
```

This will:

- Load and preprocess the data
- Train a `RandomForestClassifier`
- Print evaluation metrics
- Save the trained model to `models/final_model.pkl`

## 🧠 Model

- **Algorithm:** Random Forest
- **Metrics:** Accuracy, Precision, Recall, F1 Score

## 📦 Output

- A serialized model (`final_model.pkl`) saved to the `models/` directory
- Printed classification report in the console

## 📝 Notes

- The dataset used is expected to contain a `Survived` column and typical Titanic attributes (e.g., `Sex`, `Age`, `Pclass`, etc.).
- `PassengerId`, `Name`, `Ticket`, and `Cabin` are dropped during preprocessing.

##  License

This project is open-source and available under the MIT License.
