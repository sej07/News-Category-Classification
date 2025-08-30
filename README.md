## News Category Classification using LSTM
_This project aims to classify news articles into predefined categories using an LSTM-based neural network. The model is implemented in TensorFlow/Keras and trained on the News Category Dataset from Kaggle._

#### Dataset Details:
- Source: Kaggle(https://www.kaggle.com/datasets/rmisra/news-category-dataset/data).
- File: News_Category_Dataset_v3.json
- Contains around 200,000 news headlines and short descriptions labeled across multiple categories (e.g., Politics, Technology, Sports, Entertainment). 

#### ML Workflow: 
1. Import Libraries: 
    1. TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
2. Load Dataset
    1. Parse the .json file into a Pandas DataFrame
3. Data Preprocessing
    1. Cleaning text (removing punctuation, stopwords, lowercasing )
    2. Tokenization of text data.
    3. Padding sequences to ensure uniform input size
    4. Train-test split for supervised learning
4. Model Architecture
    - The model is built using a Recurrent Neural Network (RNN) with LSTM layers to capture sequential dependencies in text data
    1. Embedding Layer: 
        - Input Dimension: 10000 (vocabulary size)
        - Output Dimension: 100 (dense vector representation for each word)
        - Input Length: max_length (padded length of each input sequence)
    2. LSTM Layer (64 units, return_sequences=True)
    3. Dropout Layer (0.5)
    4. Second LSTM Layer (64 units)
    5. Dropout Layer (0.5)
    6. Dense Layer (64 units, ReLU activation)
    7. Output Layer (Softmax activation, num_classes units)
5. Training
    1. Loss function: Categorical Crossentropy
    2. Optimizer: Adam
    3. Batch size and epochs tuned experimentally
6. Evaluation
    1. Accuracy score on test data.
    2. Confusion matrix and classification report for per-class performance.
    3. Training/validation loss and accuracy curves for visualization.

#### Results:
1. Accuracy:0.56
2. Loss: 1.5

#### Frameworks:
- Tensorflow and Keras

#### Visualizations:
Category Comparison

<img width="978" height="646" alt="Screenshot 2025-08-30 at 3 58 22 PM" src="https://github.com/user-attachments/assets/f506522d-72d1-4644-beeb-54af94cccebd" />

Accuracy Graph

<img width="579" height="484" alt="Screenshot 2025-08-30 at 3 58 34 PM" src="https://github.com/user-attachments/assets/4e42775d-3fd1-4556-a167-9278b2e16aa5" />

Loss Graph

<img width="544" height="476" alt="Screenshot 2025-08-30 at 3 58 41 PM" src="https://github.com/user-attachments/assets/9488c8ff-6c52-41fc-b142-653435fb761e" />

#### Assumptions
1. Headlines and short descriptions contain sufficient semantic information to classify categories.
2. Padding and truncation do not distort the semantics of shorter or longer texts significantly. 

#### Improvements:
1. Deeper layer LSTM may yeild better accuracy results

#### Key Observation
1. Some categories such as "Politics" are easier to classify due to domain-specific vocabulary, while categories like "Entertainment" and "Lifestyle" often overlap, leading to misclassifications.
2. Longer text sequences improve classification accuracy but also increase training time.

