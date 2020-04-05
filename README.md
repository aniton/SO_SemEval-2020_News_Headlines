# SO_SemEval-2020_News_Headlines
Assessing the Funniness of Edited News Headlines (SemEval-2020). <br/>
For more details: [Coda Lab_(SemEval 2020 - Task 7)](https://competitions.codalab.org/competitions/20970)

## Sub-tasks

### Sub-Task 1: Regression. (the funniness for the edited headline, a real number in the 0-3 interval)
- **20 Feb 2020:** A test data release - **8 March 2020:** Submission deadline <br/>
### Sub-Task 2: Predict funnier of the two edited versions of an original headline.
- **0** - both headlines have the same funniness.
- **1**  - edited headline 1 is funnier.
- **2**  - edited headline 2 is funnier.

- **20 Feb 2020:** A test data release - **8 March 2020:** Submission deadline <br/>

## Contributors 
**Anita Soloveva**  Lomonosov MSU nit-sol@mail.ru <br/>

## Preprocessing
1. Removing ids, grades, all the following charachters  “ :. , — ˜ ”, digits and single quotation marks <br/>
2. Making a substitute  <br/>

### Sub-task 1: Approach
1. All preprocessing steps <br/>
2. Analysing whether a substitute word with its left / right context is in a list of the most frequent bigrams [14 billion word iWeb corpus](https://www.english-corpora.org/iweb/) and adding "1" or "0" to the edited headline <br/>
3. [DeepPavlov logistic regression classifier](https://github.com/aniton/SO_SemEval-2020_News_Headlines/blob/master/deeppavlov/bert.py) with [BERT English cased Embeddings](http://files.deeppavlov.ai/deeppavlov_data/bert/cased_L-12_H-768_A-12.zip)

### Sub-task 2: Approach
1. All preprocessing steps <br/>
2. Analysing whether a substitute word with its left / right context is in a list of the most frequent bigrams [14 billion word iWeb corpus](https://www.english-corpora.org/iweb/) and adding "1" or "0" to the edited headline <br/>
3. [SVM with bag of word/character n-gram features](https://github.com/aniton/SO_SemEval-2020_News_Headlines/blob/master/SVM/svm%2B.py)  <br/>
4. [Comparing](https://github.com/aniton/SO_SemEval-2020_News_Headlines/blob/master/SVM/compare.py) funiness of the same original headlines with different substitutes
