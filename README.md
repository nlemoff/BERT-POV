# POV Classification in Sentences

This project focuses on using a fine-tuned BERT model to classify sentences based on their point of view (POV). The model aims to identify and categorize text snippets into different POV classes, which is essential for tasks in literature analysis, language understanding, and narrative structure studies.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Setup](#setup)
5. [Training the Model](#training-the-model)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Files and Structure](#files-and-structure)
9. [Acknowledgments](#acknowledgments)

---

### Project Overview
This project categorizes sentences by their point of view using natural language processing (NLP) techniques. We fine-tune a BERT model to classify text based on five POV labels:

- **FP**: First-Person
- **AB**: Abstract/Background
- **TPL**: Third-Person Limited
- **TPOM**: Third-Person Omniscient
- **TPOB**: Third-Person Objective/Behavioral

By identifying the narrative POV, we can analyze texts more deeply, understanding character perspectives and narrative styles.

### Dataset
The dataset consists of labeled sentences, each assigned a specific POV label. The data is split into training, development, and testing sets:

- **train.txt**: Training data for model learning.
- **dev.txt**: Validation data for tuning model parameters.
- **test.txt**: Test data for final evaluation.

Each file contains the following columns in tab-separated format:
- **Index**: Unique identifier for each sentence.
- **Type**: All entries labeled as "adjudicated" for uniformity.
- **Label**: The POV classification label.
- **Text**: The sentence text.

#### Example
```
Index	Type	Label	Text
85	adjudicated	FP	For a moment a phrase tried to take shape in my mouth and my lips parted...
```

### Model Architecture
The core of this project is a BERT-based model with an added linear classifier for POV prediction. The architecture includes:
- **BERT Layer**: The `bert-base-cased` model, which provides contextualized embeddings.
- **Dropout Layer**: Prevents overfitting by randomly disabling neuron connections.
- **Linear Layer**: Final classifier layer to output logits corresponding to each class.

#### Code Example
```python
class EnhancedBERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(EnhancedBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, output_attentions=True)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits, outputs.attentions
```

### Setup
Clone the repository:
```bash
git clone https://github.com/nlemoff/POV-Classification-in-Sentences
```

Install dependencies:
```bash
pip install -r requirements.txt
```
Main dependencies include torch, transformers, scikit-learn, and matplotlib.

### Training the Model
Training is conducted with a custom training loop using the `train_model` function. This function optimizes the model parameters on the training data and validates it on the development set for each epoch.

```python
train_losses, train_accuracies = train_model(
    model, 
    train_data, 
    dev_data, 
    tokenizer, 
    optimizer, 
    criterion, 
    num_epochs=10, 
    batch_size=16
)
```

### Evaluation
The model is evaluated on the test set using accuracy, precision, recall, and F1-score metrics. A confusion matrix is generated to visualize model performance across POV labels.

```python
accuracy, conf_mat = evaluate_model(model, test_data, tokenizer)
print(f"Accuracy: {accuracy}")
plot_confusion_matrix(y_true, y_pred, class_names)
```

### Results
- Training Accuracy: Achieved approximately 98.7% after 10 epochs.
- Validation Accuracy: About 65% on the development set with a 95% confidence interval.
- Confusion Matrix: Displays true versus predicted labels for each class, helping to identify misclassifications.

### Files and Structure
- `AP3.ipynb`: Jupyter notebook with all code for data preparation, model training, and evaluation.
- `train.txt`, `dev.txt`, `test.txt`: The dataset files for training, validation, and testing.
- `requirements.txt`: Lists dependencies to be installed for running the code.
- `README.md`: Project documentation and instructions.

### Acknowledgments
Special thanks to the Hugging Face team for the transformers library and the PyTorch team for making deep learning accessible.
