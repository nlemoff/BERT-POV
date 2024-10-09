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
```plaintext
Index\tType\tLabel\tText
85\tadjudicated\tFP\tFor a moment a phrase tried to take shape in my mouth and my lips parted...
