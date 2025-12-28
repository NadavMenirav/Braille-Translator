# 🔤 Braille-Translator

A deep learning project that uses Convolutional Neural Networks (CNN) to recognize Braille letters from images, with a web interface powered by Anvil.

## 📋 Overview

This project implements an end-to-end machine learning pipeline that can:
- Train a CNN model to recognize individual Braille letters (A-Z)
- Process images to extract and identify multiple Braille letters
- Serve predictions through a web interface using Anvil

## 🛠️ Technologies Used

- **TensorFlow/Keras** - Deep learning framework for building and training the CNN
- **OpenCV** - Image processing and manipulation
- **NumPy** - Numerical computations and array operations
- **Pandas** - Data handling and manipulation
- **Scikit-learn** - Dataset splitting and label encoding
- **Matplotlib/Seaborn** - Visualization of training metrics
- **Anvil** - Web interface framework
- **PIL (Pillow)** - Image file handling

## 🏗️ Model Architecture

The CNN model consists of:

```
Input Layer (28x28x3 RGB images)
    ↓
Conv2D (64 filters, 5x5) + ReLU
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D
    ↓
Flatten
    ↓
Dense (576 units) + ReLU
    ↓
Dense (288 units) + ReLU
    ↓
Dense (26 units) + Softmax
```

## 🚀 Key Features

### Image Processing Pipeline
- **White margin removal** - Automatically crops unnecessary white space from images
- **Letter segmentation** - Splits images containing multiple Braille letters into individual letters
- **Image normalization** - Scales pixel values to [0, 1] range for optimal model performance

### Training Features
- **Data splitting** - 80% training, 20% testing
- **Early stopping** - Prevents overfitting by monitoring validation accuracy
- **Adam optimizer** - Adaptive learning rate for efficient convergence
- **Sparse categorical crossentropy loss** - Suitable for multi-class classification

### Web Integration
- **Anvil server connection** - Exposes model predictions via web API
- **Real-time predictions** - Accepts image uploads and returns predicted letters

## 📊 Model Performance

The model is trained for up to 6 epochs with:
- Validation split: 10% of training data
- Early stopping patience: 5 epochs
- Evaluation metrics: Accuracy and Loss tracked for both training and validation sets

## 🔧 Usage

### Training the Model

```python
# Load and preprocess images
images_list = np.array(images) / 255.0
name_list = le.fit_transform(name_list)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    images_list, name_list, test_size=0.2, random_state=42
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=6,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)
```

### Making Predictions

```python
# Single letter prediction
def printImage(image):
    letter_image = cv2.imread(image)
    letter_image = cv2.cvtColor(letter_image, cv2.COLOR_BGR2RGB)
    resized_letter = cv2.resize(letter_image, (28, 28))
    normalized_letter = resized_letter / 255.0
    input_letter = np.expand_dims(normalized_letter, axis=0)
    prediction = model.predict(input_letter)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]
```

### Web API Endpoint

```python
@anvil.server.callable
def predict_braille(image_bytes):
    # Process image and return prediction
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB').resize((28, 28))
    image_np = np.array(image) / 255.0
    image_np = np.expand_dims(image_np, axis=0)
    prediction = model.predict(image_np)
    predicted_class = np.argmax(prediction)
    predicted_letter = le.inverse_transform([predicted_class])[0]
    return predicted_letter
```

## 📁 Dataset Structure

```
Braille Dataset/
├── a1.jpg
├── a2.jpg
├── b1.jpg
├── b2.jpg
└── ... (images labeled by first character)
```

- Images are labeled by their filename's first character (e.g., 'a1.jpg' → label 'a')
- Dataset contains 26 classes (A-Z)
- Images are processed to 28x28 pixels for model input

<br><br>

This project is for educational purposes.

---

**Created with ❤️ using Python, TensorFlow, and Anvil**
