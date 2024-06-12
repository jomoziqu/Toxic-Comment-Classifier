# Toxic-Comment-Classifier

This project aims to develop a machine learning model that can classify toxic comments into six categories: toxic, severe toxicity, obscene, threat, insult, and spam. The model uses a combination of natural language processing (NLP) and deep learning techniques to accurately identify and classify toxic comments.

## Architecture

The model architecture consists of the following layers:

- Embedding Layer: The input text is converted into a sequence of embeddings using a word embedding layer with 32 dimensions.
- Bidirectional LSTM Layer: The embedding layer is fed into a bidirectional LSTM layer with 32 units, which allows the model to capture 
   both forward and backward context in the text.
- Fully Connected Feature Extractors: The output from the LSTM layer is passed through three fully connected (dense) layers with ReLU 
  activation, each with 128 units. This helps to extract relevant features from the input text.
- Output Layer: The final output layer is a dense layer with 6 units, representing the six categories of toxic comments. The sigmoid 
  activation function is used to output probabilities for each category.
  
## Training and Evaluation

The model is trained on a dataset of labeled toxic comments and evaluated using metrics such as accuracy, precision, recall, and F1-score. The hyperparameters used in the model are optimized using a combination of grid search and cross-validation.

## Dependencies

This project requires the following dependencies:

- TensorFlow (for building and training the model)
- Keras (for building and training the model)
- NumPy (for numerical computations)
- Pandas (for data manipulation and analysis)
- Scikit-learn (for preprocessing and evaluation)
- Gradio (for UI)

## AUTHOR
# WILSON JOMO
