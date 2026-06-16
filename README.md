# Iris_data_classification

# 🌸 Iris Flower Classification using Deep Learning

## 📌 Overview

This project uses a Deep Learning Neural Network built with TensorFlow/Keras to classify Iris flowers into three species:

- Setosa
- Versicolor
- Virginica

The model is trained on the famous Iris dataset and predicts the flower species based on four flower measurements.

---

## 📊 Dataset

The Iris dataset is loaded directly from the Seaborn library.

```python
import seaborn as sns

df = sns.load_dataset('iris')
```

### Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Target Classes

| Species | Encoded Label |
|----------|----------|
| Setosa | 0 |
| Versicolor | 1 |
| Virginica | 2 |

---

## ⚙️ Data Preprocessing

Convert categorical species names into numerical labels.

```python
dictionary = {
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
}

df['species'] = df['species'].map(dictionary)
```

Create feature and target datasets:

```python
x = df.drop('species', axis=1)

import tensorflow as tf

y = tf.keras.utils.to_categorical(df['species'])
```

---

## 🧠 Model Architecture

The neural network consists of:

| Layer | Neurons |
|---------|---------|
| Input Layer | 4 |
| Hidden Layer 1 | 32 |
| Hidden Layer 2 | 32 |
| Hidden Layer 3 | 64 |
| Hidden Layer 4 | 64 |
| Hidden Layer 5 | 128 |
| Hidden Layer 6 | 64 |
| Output Layer | 3 (Softmax) |

```python
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(4, input_dim=4))
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(3, activation='softmax'))
```

---

## 🚀 Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 🎯 Model Training

Train the model for 50 epochs.

```python
model.fit(x, y, epochs=50)
```

---

## 🔍 Sample Predictions

### Example 1

Input:

```python
np.array([[5.1, 3.5, 1.4, 0.2]])
```

Prediction:

```python
[[9.9992085e-01, 7.9167861e-05, 3.6665598e-20]]
```

Predicted Species:

```text
Setosa
```

---

### Example 2

Input:

```python
np.array([[6.7, 3.0, 5.2, 2.3]])
```

Prediction:

```python
[[2.8113180e-07, 5.7436787e-02, 9.4256288e-01]]
```

Predicted Species:

```text
Virginica
```

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Seaborn
- TensorFlow
- Keras

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification
```

Install required libraries:

```bash
pip install numpy pandas seaborn tensorflow
```

---

## ▶️ Run the Project

```bash
python iris_classification.py
```

---

## 📁 Project Structure

```text
iris-flower-classification/
│
├── iris_classification.py
├── README.md
└── requirements.txt
```

---

## 🔮 Future Enhancements

- Train-Test Split
- Data Normalization
- Accuracy and Loss Visualization
- Confusion Matrix
- Hyperparameter Tuning
- Model Saving and Loading
- Flask Deployment
- Streamlit Web Application

---

## 📈 Results

The Deep Learning model successfully learns the patterns in the Iris dataset and accurately classifies flowers into their respective species based on sepal and petal measurements.

---

## 👨‍💻 Author

**Grandhi Lahari**

If you found this project useful, feel free to ⭐ the repository.
