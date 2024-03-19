## Basic strategy overview:

* For each riddle, given its length n, letters will be guessed according to two possible schemes:

  * If no letter has been correctly guessed yet, letters will be guessed based on the frequency of letters in all n-long words in the dictionary.
  
  * If there have already been some correct guesses, an RNN model will be used to predict the next letter to be guessed.

## Table of contents.

### 1. Loading the existing dataset as a list.

### 2. Creating a dictionary of most common letters in the words of each word length.

### 3. Create dataset for training

In order to train the RNN model that follows, synthetic riddles and their respective correct answers will be needed. These will be created according to the steps outlined below:

### 3.1. Creating synthetic riddles

For each word w in the training vocabulary, composed of a set k of unique letters, we can generate potential riddles. This is done by masking a subset of letters in k, where this subset is an element of the power set of k. Consequently, each element k_i from the power set of k represents a possible riddle, when all the letters in k_i are concealed in the riddle. For each riddle generated, a corresponding guess containing all the omitted letters is created, representing all the possible correct guesses. For example, in one iteration, a riddle "Co__ect_c_t" and the correct guess "niu" would be created.

The function `hangman_guesses(word,n)` below creates riddles from word word, limited to n riddles to avoid an excessive bias towardss words composed of a large number of different letters.  The cardinality of the power set being 2^{len(k)} means that without this limitation the RNN's exposure to shorter words would be restricted. In the example below, n=64.

Finally, a subset of 15% of all riddles and guesses were used for the following steps, to ensure feasible training times. This network was trained in an Apple M1 Pro 2021, 8-core, with 16GB RAM.

### 3.2. Preprocessing the data for the RNN

#### 3.2.1. Preprocessing riddles

Each riddle underwent TextVectorization, converting each of the possible characters in the riddle to an integer. Each letter is an object of interest, so riddles were split by character. Riddle length was limited to 15 characters to accommodate the majority of the words in the training set.

#### 3.2.2. Preprocessing guesses

This stage maps each guess, e.g. "eiu" in "Conn_ct_c_t" to a 1 x 26 vector of probabilities (here, [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]), corresponding to the 26 letters of the alphabet. This encoding employs `tf.keras.layers.StringLookup` for multi-hot encoding.

The following steps involved:

- Removing the first element from each encoded guess, which corresponds to the "Unknown" character. This character is present in all encoded guesses and does not contribute information to the RNN.

- Converting both guesses and riddles into tensor slices.

Each element of the final dataset is a 2-tuple with an encoded riddle and an encoded guess. Samples are processed in 64-unit batches. The prefetching of one batch is added to improve performance.

Finally, the dataset is split into a train-set and a test-set, containing respectively 80% and 20% of the data. 

This setup was demonstrated using the following code snippet:

## 4. RNN model

An RNN model is constructed to estimate the best guess for each riddle.

- The number of tokens is set to 28, which is equal to the 26 letters in the English language and two additional tokens (underscore sign and "unknown")

- Initially, the RNN model is composed of the 256-cell long bidirectional recurrent layers. These bidirectional layers capture dependencies in both directions of the input sequence, which is important to understand the context around missing letters in hangman. 

- The recurrent dropout of 0.1 in both GRU layers and the subsequent dropout of 0.1 in the following layer are used to prevent overfitting, thus allowing for better performance out of sample (which is especially important since hangman will be played with unknown words)

- The dense layer of 128 units and ELU activation provides nonlinearity to the model and helps to learn complex patterns from the data. ELU handles the vanishing gradient problem faster than ReLU and shows faster convergence in the data as well.

- The final Dense layer has 26 units, corresponding to the 26 letters of the alphabet, with softmax activation to output a probability distribution over all possible letters.  

- The model is initially set to run through a maximum of 100 epochs. EarlyStopping is employed to prevent overfitting by halting training if the model's loss doesn't improve after three epochs, starting from epoch 2, restoring the weights from its best-performing iteration.

- The model is optimized using the Nadam optimizer and binary crossentropy loss; which will be minimized as the vector of predicted probabilities approaches the vector of true probabilities mentioned in Section 3.2.2. The precision metric is configured to evaluate the model's performance, specifically measuring the precision of the prediction with highest probability, which is critical in a game like hangman where the first guess significantly influences game progress.

- Several hyperparameters were calibrated (number of cells in each layer; number of layers; dropout rates). Training time were also considered.

- The model was fit with the trainset, whereas performance is evaluated in unseen riddles (`test_dataset`), since generalizing well to unseen data is fundamental here. The number of `steps_per_epoch` was limited to 25,000 for increased computation speed.
