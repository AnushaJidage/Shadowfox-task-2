# 📊 Machine Learning Training Pipeline

This project is a complete end-to-end machine learning pipeline built in Python. It performs:

* Data loading (with error handling)
* Exploratory Data Analysis (EDA)
* Data preprocessing
* Model training
* Model evaluation
* Saving trained models and encoders

---

## 🚀 Features

* ✅ Automatic dataset detection if file not found
* ✅ Handles missing values
* ✅ Encodes categorical variables
* ✅ Generates EDA visualizations
* ✅ Trains a regression model
* ✅ Saves model and encoders for reuse
* ✅ Beginner-friendly and robust

---

## 📁 Project Structure

```
project/
│── train_model.py
│── data.csv (your dataset)
│── outputs/
│   ├── heatmap.png
│   ├── histograms.png
│   ├── model.pkl
│   └── encoders.pkl
```

---

## ⚙️ Installation

Make sure you have Python 3.8+ installed.

Install required libraries:

```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## ▶️ Usage

1. Place your dataset in the project folder
2. Rename it to:

```
data.csv
```

OR update the file path inside the script:

```python
file_path = "your_file.csv"
```

3. Run the script:

```
python train_model.py
```

---

## 📊 Output

After running, an `outputs/` folder will be created containing:

* 📈 `heatmap.png` → Correlation heatmap
* 📊 `histograms.png` → Feature distributions
* 🤖 `model.pkl` → Trained machine learning model
* 🔤 `encoders.pkl` → Label encoders for categorical data

---

## 🧠 Model Details

* Algorithm: Linear Regression
* Train/Test Split: 80/20
* Metrics:

  * Mean Squared Error (MSE)
  * R² Score

---

## ⚠️ Notes

* The script automatically selects the **last column as the target variable**
* All categorical columns are label encoded
* Only numeric columns are used for correlation heatmaps
* Large datasets may take longer to process

---

## 🛠️ Troubleshooting

### File not found error

Make sure your dataset is in the same folder or update the path:

```python
file_path = r"C:\full\path\to\your\data.csv"
```

---

### No plots showing

Plots are saved in the `outputs/` folder instead of displaying on screen.

---

### Model not saving

Check if the `outputs/` folder exists and you have write permissions.

---

## 📌 Future Improvements

* Add classification models
* Hyperparameter tuning
* Feature selection
* Model comparison
* Web deployment (Flask / FastAPI)

---

## 📄 License

This project is open-source and free to use.

---

## 🙌 Acknowledgments

Built using:

* pandas
* NumPy
* scikit-learn
* matplotlib
* seaborn

---

## 💡 Author

Your Name
(You can add your GitHub profile here)
