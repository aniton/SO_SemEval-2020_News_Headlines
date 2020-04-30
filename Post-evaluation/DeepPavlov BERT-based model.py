from deeppavlov.dataset_readers.basic_classification_reader import BasicClassificationDatasetReader
from deeppavlov.dataset_iterators.basic_classification_iterator import BasicClassificationDatasetIterator
from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.models.preprocessors.one_hotter import OneHotter
from deeppavlov.models.classifiers.proba2labels import Proba2Labels
from deeppavlov.models.bert.bert_classifier import BertClassifierModel
from deeppavlov.metrics.accuracy import sets_accuracy
reader = BasicClassificationDatasetReader()
data = reader.read(data_path="./stanfordSentimentTreebank", 
                   train="/content/train.csv", valid="/content/valid.csv", test="/content/test1.csv",
                   x="original", y="meanGrade")
iterator = BasicClassificationDatasetIterator(data, seed=42, shuffle=True)
bert_preprocessor = BertPreprocessor(vocab_file="~/.deeppavlov/downloads/bert_models/cased_L-12_H-768_A-12/vocab.txt",
                                     do_lower_case=False,
                                     max_seq_length=64)
vocab = SimpleVocabulary(save_path="./binary_classes.dict")
iterator.get_instances(data_type="train")
vocab.fit(iterator.get_instances(data_type="train")[1])
one_hotter = OneHotter(depth=vocab.len, 
                       single_vector=True  # means we want to have one vector per sample
                      )
prob2labels = Proba2Labels(max_proba=True)
bert_classifier = BertClassifierModel(
    n_classes=vocab.len,
    return_probas=True,
    one_hot_labels=True,
    bert_config_file="~/.deeppavlov/downloads/bert_models/cased_L-12_H-768_A-12/bert_config.json",
    pretrained_bert="~/.deeppavlov/downloads/bert_models/cased_L-12_H-768_A-12/bert_model.ckpt",
    save_path="sst_bert_model/model",
    load_path="sst_bert_model/model",
    keep_prob=0.5,
    learning_rate=0.5,
    learning_rate_drop_patience=5,
    learning_rate_drop_div=2.0
)
# Method `get_instances` returns all the samples of particular data field
x_valid, y_valid = iterator.get_instances(data_type="test")
# You need to save model only when validation score is higher than previous one.
# This variable will contain the highest accuracy score
best_score = 0.
patience = 2
impatience = 0

# let's train for 3 epochs
for ep in range(3):
  
    nbatches = 0
    for x, y in iterator.gen_batches(batch_size=256, 
                                     data_type="train", shuffle=True):
        x_feat = bert_preprocessor(x)
        y_onehot = one_hotter(vocab(y))
        bert_classifier.train_on_batch(x_feat, y_onehot)
        print("Batch done\n")
        nbatches += 1
        
        if nbatches % 1 == 0: 
            y_valid_pred = bert_classifier(bert_preprocessor(x_valid))
            score = sets_accuracy(y_valid, vocab(prob2labels(y_valid_pred)))
            
    y_valid_pred = bert_classifier(bert_preprocessor(x_valid))
    score = sets_accuracy(y_valid, vocab(prob2labels(y_valid_pred)))
    print("Epochs done: {}. Valid Accuracy: {}".format(ep + 1, score))
    if score > best_score:
        bert_classifier.save()
        best_score = score    
        impatience = 0
    else:
        impatience += 1
        if impatience == patience:
            break

for a in y_valid_pred:
    f5.write(a + "\n")
    print(a + "\n")
