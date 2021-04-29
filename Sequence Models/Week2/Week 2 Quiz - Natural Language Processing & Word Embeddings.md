## Week 2 Quiz - Natural Language Processing & Word Embeddings


#### 1. Suppose you learn a word embedding for a vocabulary of 10000 words. Then the embedding vectors should be 10000 dimensional, so as to capture the full range of variation and meaning in those words.
##### Ans: False
#### 2. What is t-SNE?
##### Ans: A non-linear dimensionality reduction technique
#### 3. Suppose you download a pre-trained word embedding which has been trained on a huge corpus of text. You then use this word embedding to train an RNN for a language task of recognizing if someone is happy from a short snippet of text, using a small training set.
```
x (input text)			y (happy?)
I'm feeling wonderful today!	1
I'm bummed my cat is ill.	0
Really enjoying this!		1
```
#### Then even if the word �ecstatic� does not appear in your small training set, your RNN might reasonably be expected to recognize �I�m ecstatic� as deserving a label y=1y = 1y=1.
##### Ans: True
#### 4. Which of these equations do you think should hold for a good word embedding? (Check all that apply) 
##### Ans: 
- e_{boy} - e_{girl} \approx e_{brother} - e_{sister}
- e_{boy} - e_{brother} \approx e_{girl} - e_{sister}
#### 5. Let E be an embedding matrix, and let o_{1234}? be a one-hot vector corresponding to word 1234. Then to get the embedding of word 1234, why don�t we call ``E * o_{1234}```? in Python?
##### Ans: It is computationally wasteful.
#### 6. When learning word embeddings, we create an artificial task of estimating P(target | context). It is okay if we do poorly on this artificial prediction task; the more important by-product of this task is that we learn a useful set of word embeddings. 
##### Ans: True
#### 7. In the word2vec algorithm, you estimate P(t | c), where t is the target word and c is a context word. How are t and c chosen from the training set? Pick the best answer.
##### Ans: c and t are chosen to be nearby words.
#### 8. Suppose you have a 10000 word vocabulary, and are learning 500-dimensional word embeddings. The word2vec model uses the following softmax function:
```tex
P(t \mid c) = \frac{e^{\theta_t^T e_c}}{\sum_{t�=1}^{10000} e^{\theta_{t�}^Te_c}}
```
#### Which of these statements are correct? Check all that apply.
##### Ans: 
- \theta_t and e_c are both 500 dimensional vectors.
- \theta_t and e_c are both trained with an optimization algorithm such as Adam or gradient descent. 
#### 9. Suppose you have a 10000 word vocabulary, and are learning 500-dimensional word embeddings.The GloVe model minimizes this objective:
```tex
\min \sum_{i=1}^{10,000} \sum_{j=1}^{10,000} f(X_{ij}) (\theta_i^T e_j + b_i + b_j� - log X_{ij})^2
```
#### 
##### Ans: 
- \theta_i and e_j should be initialized randomly at the beginning of training.
- X_{ij} is the number of times word i appears in the context of word j.
- The weighting function f(.) must satisfy f(0) = 0. 
#### 10. You have trained word embeddings using a text dataset of m_1? words. You are considering using these word embeddings for a language task, for which you have a separate labeled dataset of m_2? words. Keeping in mind that using word embeddings is a form of transfer learning, under which of these circumstance would you expect the word embeddings to be helpful?
##### Ans: m_1 >> m_2
