from os import listdir
from os.path import join, splitext

from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC

def run_model(path_dataset, norm_, idf_, smooth_, subl_, min_df_, max_df_):

    print('Processing directory:', path_dataset)
    
    train_scores = []
    train_labels = []
    score_names = []
    
    # read data
    for candidate in listdir(path_dataset):
        #print('Candidate:', candidate)
        
        path_candidate = join(path_dataset, candidate)
        for score in listdir(path_candidate):
            
            path_score = join(path_candidate, score)
            with open(path_score, 'r') as f:
                train_scores.append(f.read())
                train_labels.append(candidate)
                score_names.append(score.split('.')[0])
    
    result = my_loo(score_names, train_scores, train_labels, norm_, idf_, smooth_, subl_, min_df_, max_df_)
    print('min_df', min_df_, '\t', 'max_df', max_df_, '\t', 'norm', norm_, '\t', 'use_sublinear', \
    subl_, '\t', 'use_idf', idf_, '\t ', 'use_smooth', smooth_, '\t ', 'result', result, '\n')

def my_loo(score_names, train_scores, train_labels_, norm_, idf_, smooth_, subl_, min_df_, max_df_):

    sum_feats = 0
    counter = 0
    for i in range(len(train_scores)):

        test_data = [train_scores[i]]
        test_label = [train_labels_[i]]
        test_score_name = score_names[i]

        train_data = []
        train_labels = []

        for j in range(len(train_scores)):
            if j != i:
                train_data.append(train_scores[j])
                train_labels.append(train_labels_[j])

        tokenizer_ = RegexpTokenizer(r'[\n]+', gaps=True)
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), lowercase=False, \
                tokenizer=tokenizer_.tokenize, min_df=min_df_, max_df=max_df_, binary=False)
        transformer = TfidfTransformer(norm=norm_, use_idf=idf_, smooth_idf=smooth_, sublinear_tf=subl_)

        train_data = vectorizer.fit_transform(train_data)
        test_data = vectorizer.transform(test_data)

        train_data = transformer.fit_transform(train_data)
        test_data = transformer.transform(test_data)

        train_data = train_data.toarray()
        test_data = test_data.toarray()

        number_feats = len(vectorizer.get_feature_names())
        sum_feats += number_feats
        
        model = SVC(kernel='linear', C=1000)
        model.fit(train_data, train_labels)
        predictions=model.predict(test_data)

        if predictions == test_label:
            counter += 1
        else:
            print(test_score_name)

    print('Average of features', round((sum_feats/107), 0))
    
    return round((counter*100/107), 2) 

def run_all():
    # Best from MIDI representation
    norm_ = 'l2'
    idf_ = True
    smooth_ = True
    subl_ = True
    min_df_ = 0
    max_df_ = 97
    path_dataset = join('long_rep', 'midi_pitch', '11. 1V-2V-V')
    run_model(path_dataset, norm_, idf_, smooth_, subl_, min_df_, max_df_)
    
    # Best from class representation
    norm_ = 'l2'
    idf_ = True
    smooth_ = False
    subl_ = True
    min_df_ = 5
    max_df_ = 107
    path_dataset = join('long_rep', 'class_pitch', '15. 1V-2V-V-C')
    run_model(path_dataset, norm_, idf_, smooth_, subl_, min_df_, max_df_)
    
run_all()