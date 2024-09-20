# README

## Artificial Neural Networks - 5th Assignment

This assignment explores various concepts in **Recurrent Neural Networks (RNNs)** and their applications, specifically focusing on the use of LSTM-based models for Persian language modeling.

### **Table of Contents**
1. [Question 1 - Stateful RNN vs Stateless RNN](#question-1)
2. [Question 2 - Encoder-Decoder RNN vs Plain Sequence-to-Sequence RNN](#question-2)
3. [Question 3 - Designing Gated RNNs for Summing Inputs](#question-3)
4. [Question 5 - Persian Language Model Implementation](#question-5)
5. [How to Run the Code](#how-to-run)

---

### **Question 1 - Stateful RNN vs Stateless RNN** <a name="question-1"></a>

**Stateful RNN** and **Stateless RNN** differ in how they handle the hidden state across batches of sequences:
---

### **Question 2 - Encoder-Decoder RNN vs Plain Sequence-to-Sequence RNN** <a name="question-2"></a>

**Encoder-Decoder RNN** and **Plain Sequence-to-Sequence RNN** models are used for tasks like machine translation, but they differ in their structure and function:
---

### **Question 3 - Designing Gated RNNs for Summing Inputs** <a name="question-3"></a>

For this question, the goal is to design a gated RNN cell that sums its inputs over time by adjusting the gating values.
---

### **Question 5 - Persian Language Model Implementation** <a name="question-5"></a>

The implementation of the Persian language model is based on LSTM cells for sequence prediction. The dataset used for training is a subset of the Persian Wikipedia dataset. The goal is to predict the next characters in a sentence, given a sequence of input characters (3 to 5 words).

#### **Implementation Steps**:

1. **Dataset Loading**: 
   - The Persian Wikipedia text dataset is loaded into the system, and the data is preprocessed to remove digits and normalize the text.

2. **Character Indexing**: 
   - The model works at the character level, so we create mappings from characters to integer indices (`char_to_int`) and from integers back to characters (`int_to_char`).
   - The text is split into sequences of 60 characters each, with corresponding target characters representing the next character in the sequence.

3. **Model Architecture**:
   - The LSTM model consists of two stacked LSTM layers with 128 units each, followed by dropout layers to reduce overfitting.
   - The final layer is a dense layer with a softmax activation function, predicting the probability distribution of the next character.

4. **Training**: 
   - The model is trained for 50 epochs using the **RMSprop** optimizer and **categorical cross-entropy** as the loss function.
   - We monitor the loss during training and save the model weights for later use.

5. **Perplexity Calculation**:
   - Perplexity is used to evaluate the performance of the model. It is a standard metric for language models and represents how well the model predicts the next character in the sequence. The lower the perplexity, the better the model's generalization.

6. **Text Generation**:
   - The trained model can generate text by taking a starting string of 3 to 5 words and predicting the next characters in the sequence.
   - We use a sampling function to sample the next character based on the probability distribution predicted by the model.

7. **Perplexity Measurement**:
   - Perplexity is calculated on the test dataset to evaluate the model's generalization performance.

---

### **How to Run the Code** <a name="how-to-run"></a>

1. Install required libraries:
   ```bash
   pip install tensorflow keras numpy
   ```

2. Place the Persian Wikipedia dataset in the input directory, and ensure the paths are correctly referenced in the code.

3. Run the code for training the model:
   ```bash
   python train_model.py
   ```

4. To generate text, use the `text_prediction()` function with a starting sequence and the trained model weights.

---
