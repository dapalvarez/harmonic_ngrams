from os import listdir
from os.path import join, splitext

from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC

def run_models(path_dataset):

    print('Processing directory:', path_dataset)
    
    train_scores = []
    train_labels = []
    
    # read data
    for candidate in listdir(path_dataset):
        #print('Candidate:', candidate)
        
        path_candidate = join(path_dataset, candidate)
        for score in listdir(path_candidate):
            
            path_score = join(path_candidate, score)
            with open(path_score, 'r') as f:
                train_scores.append(f.read())
                train_labels.append(candidate)
    
    norms = ['l1', 'l2']
    idf = [False, True]
    smooth = [False, True]
    sublin = [False, True]
    min_dfs = [0, 5, 10]
    max_dfs = [len(train_labels), len(train_labels) - 5, len(train_labels) - 10]
    
    # do grid search
    for mind in min_dfs:
        for maxd in max_dfs:
            for nor in norms:
                for sl in sublin:
                    for df in idf:
                        for sm in smooth:
                            result = my_loo(train_scores, train_labels, nor, df, sm, sl, mind, maxd)
                            print('min_df', mind, '\t', 'max_df', maxd, '\t', 'norm', nor, '\t', 'use_sublinear', \
                            sl, '\t', 'use_idf', df, '\t ', 'use_smooth', sm, '\t ', 'result', result, '\n')

def my_loo(train_scores, train_labels_, norm_, idf_, smooth_, subl_, min_df_, max_df_):

    sum_feats = 0
    counter = 0
    for i in range(len(train_scores)):

        test_data = [train_scores[i]]
        test_label = [train_labels_[i]]

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

    print('Average of features', round((sum_feats/107), 0))
    
    return round((counter*100/107), 2) 

def run_all():
    representation_types = ['midi_pitch', 'class_pitch']
    
    for rep_type in representation_types:
        path_dataset = join('long_rep', rep_type)
        
        for dataset in listdir(path_dataset):
            run_models(join(path_dataset, dataset))
    
run_all()