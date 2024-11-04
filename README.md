# CS441 : Homework 2

### Author : Akhil S Nair
### Email : anair56@uic.edu

### Youtube video : https://youtu.be/te6ZUrHZflo?si=IIYvIn7rQdLV3ucM
(The video explains the deployment of the Spark application in the AWS EMR Cluster and the project structure.)

## Overview

The application reads token IDs from a text file, generates embeddings, and trains a neural network model using a sliding window technique. It utilizes *Apache Spark* for distributed data processing and DL4J for deep learning operations.

## Environment
- **OS:** Mac
- **IDE:** IntelliJ IDEA 2023.3.8 (Ultimate Edition)
- **Scala Version:** 2.12.18
- **SBT Version:** 1.10.1
- **Spark Version:** 3.4.3
- **Apache Hadoop:** (version 3.3.4)
- **Java JDK:** (Java 11 )

## Getting Started

#Youtube video Link: [https://youtu.be/te6ZUrHZflo?si=IIYvIn7rQdLV3ucM](https://youtu.be/te6ZUrHZflo?si=IIYvIn7rQdLV3ucM)


## Components

1. **SlidingWindowSpark**:
   - Processes token sequences using a sliding window approach, generating contextual embeddings with positional encoding.
   - Loads embeddings and token datasets, applies a configurable sliding window to generate context, and exports processed data for training LLMs.
   - Handles data loading and positional embedding calculations, enabling a rich representation for each token's context.

2. **TrainingService**:
   - Facilitates the distributed training of embeddings using Apache Spark and Deeplearning4j.
   - Initializes and configures a multi-layer neural network (LSTM) for sequence modeling.
   - Manages the entire training pipeline, including data sharding, distributed processing, and metrics logging.
   - Stores the trained model and training statistics to HDFS for further use in downstream LLM tasks.


### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/akhilmw2/CS441_Fall2024_Assignment_2.git
   cd CS441_Fall2024_Assignment_2


2. **Run the Main Classes**

   To execute the pipeline, run the main classes:
   
   For preprocessing and embedding generation: SlidingWindowSpark
   For training: TrainingService
   The main classes will handle data processing and model training.


3. **Output Files**

   After running the main classes, you will find the output files:
   ```bash
      Statistics: src/main/resources/stats.txt
      Model: src/main/resources/trainedModel.zip

## Local Development Commands

### Running Tests
Test files can be found under the directory `src/test`:
```bash
sbt clean compile test
```

### Local Execution
```bash
sbt clean compile run
```

## Conclusion
In this project, we used Apache Spark and Deeplearning4j (DL4J) to implement a sliding window approach on embeddings and calculate positional embeddings. We then trained a neural network model using DL4J to generate and refine these embeddings, leveraging Spark's distributed processing to handle large datasets efficiently.

This work demonstrates the application of distributed computing techniques to modern NLP tasks, showcasing how Spark and DL4J can be combined for large-scale, cloud-based processing of embedding and positional data.

## Acknowledgments
I would like to thank Professor Mark Grechanik for his guidance and instruction in the CS 441 course, which has been invaluable in developing and refining this project. Special thanks to TA Vasu Garg for support and insights throughout the process.




