import pickle
from statistics import mean, mode
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
import utils
import numpy as np
import os
import matplotlib.pyplot as plt

def _metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)

    tn, tp = tp, tn
    fn, fp = fp, fn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    

    print(f'| tp={tp} | fn={fn} | tn={tn} | fp={fp} |\n')
    print(f'| precision={precision:.2%} | recall={recall:.2%} | f1={f1:.2%} | acc={acc:.2%} |\n')

def _multi_metrics(config, y_true, y_pred):
    fam2idx, idx2fam = utils.pickle_load('./Analysis/VT_NameMap/resource/fam2idx.pkl')
    accept_label = [fam2idx[item] for item in config.multi_classes]

    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=accept_label)

    label_result = dict({})
    for i in range(mcm.shape[0]):
        label = idx2fam[accept_label[i]]
        label_result[label] = {}
        cm = mcm[i]
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        label_result[label]['tn'] = tn 
        label_result[label]['fp'] = fp 
        label_result[label]['fn'] = fn 
        label_result[label]['tp'] = tp

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # print(f'| tn={tn} | fp={fp} | fn={fn} | tp={tp} |\n')
    print(f'| precision={precision:.2%} | recall={recall:.2%} | f1={f1:.2%} | acc={acc:.2%} |\n')



def _model(config, data_loaders, model):
    X, y = data_loaders

    scoring = {'acc': 'accuracy',
            'prec_macro': 'precision_macro',
            'rec_micro': 'recall_macro'}
    scores = cross_validate(model, X, y, scoring=scoring, cv=5, return_train_score=True, return_estimator=True)
    print(scores.keys())
    print(scores['test_acc']) 
    print(f"Mean score: {mean(scores['test_acc'])}")

def _train_test_split(config, X, y, names, test_size=0.3, random_state=3, train_cut_index=None):

    if random_state:
        np.random.seed(random_state)

    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    train_index = shuffled_indexes[test_size:]
    test_index = shuffled_indexes[:test_size]
    # train_start_year, train_end_year = config.train_start_year, config.train_end_year
    # test_start_year, test_end_year = config.test_start_year, config.test_end_year
    # accept_train_years = list(range(train_start_year, train_end_year + 1))
    # accept_test_years = list(range(test_start_year, test_end_year + 1))

    # train_index = []
    # test_index = []
    # for i, name in enumerate(names):
    #     name = name.split('\\')[-1].split('.')[0]
    #     if(config.name2time[name]['year'] not in accept_train_years + accept_test_years): continue
    #     if(config.name2time[name]['year'] in accept_train_years): train_index.append(i)
    #     elif(config.name2time[name]['year'] in accept_test_years): test_index.append(i)
    #     else: raise("YEAR ERROR")
    
    train_X = [X[i] for i in train_index]
    train_y = [y[i] for i in train_index]
    train_names = [names[i] for i in train_index]
    test_X = [X[i] for i in test_index]
    test_y = [y[i] for i in test_index]
    test_names = [names[i] for i in test_index]
    
    return train_X, test_X, train_y, test_y, train_names, test_names

def write_wrong_sample(config, y_true, y_pred, names):
    target_indexes = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    filtered_names = [ names[i] for i in target_indexes ]
    black_count = len(list(filter(lambda name: 'black' in name, filtered_names)))
    white_count = len(filtered_names) - black_count
    save_path = os.path.join(config.wrong_sample_output_dir, f'{config.model_name}-{config.data_name}-{config.pid_flag}[{config.api_count_min}-{config.sequence_length_min}]' + (f'[{config.rw_k}-{config.rw_step}]' if config.pid_flag == 'pid' else '') + f'-{config.epochs}-{config.vector_size}-{config.model_type}-{config.data_type}.html')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f'<h3>Black Count: {black_count}; White Count: {white_count}</h3><br>')
        for name in filtered_names:
            f.write(f'<a href="file:///{name}">{name}</a><br>')

def write_black_sample(CONFIG, y_true, y_pred, names):
    save_dir = './outputs/black_family'
    save_path = os.path.join(save_dir, CONFIG.black_family_name)
    pred_black_names = dict({
        "true": [],
        "false": []
    })
    for i in range(len(y_true)):
        if(y_true[i] == 1): continue
        if(y_pred[i] == y_true[i]):
            pred_black_names['true'].append(names[i])
        else:
            pred_black_names['false'].append(names[i])
    with open(save_path, 'wb') as f:
        pickle.dump(pred_black_names, f)

def gnb(config, data_loaders):
    model = GaussianNB()
    _model(config, data_loaders, model)
def mnb(config, data_loaders):
    model = MultinomialNB()
    _model(config, data_loaders, model)
def gbdt(config, data_loaders):
    X, y = data_loaders
    
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=3)

    e_range = [ 2 ** i for i in range(1, 9)]
    # e_range = [ i for i in range(2, 10)]
    scores = []

    for e in e_range:
        model = GradientBoostingClassifier(n_estimators=e)

        cv_scores = cross_val_score(model, train_X, train_y, cv=5, scoring='accuracy')
        scores.append(cv_scores.mean())

        print(f'e = {e}, mean_score = {cv_scores.mean()}\nscores = {cv_scores}')

    utils.show_plot(e_range, scores, 'e', 'accuracy', save=False)

    
    best_scores_index = scores.index(max(scores))
    print(f"scores = {scores}")
    print(f"best score = {scores[best_scores_index]}")
    print(f"best e = {e_range[best_scores_index]}")
def rsvc(config, data_loaders):
    model = SVC()
    _model(config, data_loaders, model)
def lsvc(config, data_loaders):
    model = LinearSVC()
    _model(config, data_loaders, model)
def psvc(config, data_loaders):
    model = SVC(kernel='poly')
    _model(config, data_loaders, model)
def ssvc(config, data_loaders):
    model = SVC(kernel='sigmoid')
    _model(config, data_loaders, model)
def lr(config, data_loaders):
    model = LogisticRegression(max_iter=9000)
    _model(config, data_loaders, model)

def dtree(config, data_loaders):
    """_result_

        * nopid: [0.93658159 0.94225316 0.9352926  0.94070637 0.94146467]
        * pid: [0.94096417 0.94019077 0.936066   0.94070637 0.94223827]
    """
    X, y = data_loaders
    train_X, test_X, train_y, test_y, train_names, test_names = _train_test_split(X, y, test_size=0.3, random_state=3, train_cut_index=train_cut_index)

    model = tree.DecisionTreeClassifier(criterion="entropy")
    scoring = {'acc': 'accuracy',
            'prec_macro': 'precision_macro',
            'rec_micro': 'recall_macro'}
    scores = cross_validate(model, train_X, train_y, scoring=scoring, cv=5, return_train_score=True, return_estimator=True)
    print(scores.keys())
    print(scores['test_acc']) 
    print(f"Mean score: {mean(scores['test_acc'])}")
    model.fit(train_X, train_y)
    y_pred= model.predict(test_X)
    _metrics(test_y, y_pred)
    write_wrong_sample(config, test_y, y_pred, test_names)

def knn(config, data_loaders):

    ((train_X, train_y, train_names), (test_X, test_y, test_names)) = data_loaders

    # k_range = [ 2 ** i for i in range(1, 7)]
    k_range = [ i for i in range(2, 10)]
    scores = []

    for k in k_range:
        knn = KNeighborsClassifier(k)
        cv_scores = cross_val_score(knn, train_X, train_y, cv=10, scoring='accuracy')
        scores.append(cv_scores.mean())

        print(f'k = {k}, mean_score = {cv_scores.mean()}\nscores = {cv_scores}')

    utils.show_plot(k_range, scores, 'k', 'accuracy', save=False)

    
    best_mean_scores_index = scores.index(max(scores))
    best_k = k_range[best_mean_scores_index]
    print(f"scores = {scores}")
    print(f"best mean score = {scores[best_mean_scores_index]}")
    print(f"best k = {best_k}")

    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(train_X, train_y)
    y_pred = best_knn.predict(test_X)
    _metrics(test_y, y_pred)
    # write_wrong_sample(config, test_y, y_pred, test_names)
    # write_black_sample(config, test_y, y_pred, test_names)
    print(best_knn.score(test_X, test_y))

def rf(config, data_loaders):
    X, y = data_loaders
    
    train_X, test_X, train_y, test_y, train_names, test_names = _train_test_split(config, X, y, test_size=0.3, random_state=3, train_cut_index=train_cut_index)

    e_range = [ 2 ** i for i in range(1, 9)]
    # e_range = [ i for i in range(2, 10)]
    scores = []

    for e in e_range:
        model = RandomForestClassifier(n_estimators=e)

        cv_scores = cross_val_score(model, train_X, train_y, cv=5, scoring='accuracy')
        scores.append(cv_scores.mean())

        print(f'e = {e}, mean_score = {cv_scores.mean()}\nscores = {cv_scores}')

    utils.show_plot(e_range, scores, 'e', 'accuracy', save=False)

    
    best_scores_index = scores.index(max(scores))
    best_e = e_range[best_scores_index]
    print(f"scores = {scores}")
    print(f"best score = {scores[best_scores_index]}")
    print(f"best e = {best_e}")

    best_rf = RandomForestClassifier(n_estimators=best_e)
    best_rf.fit(train_X, train_y)
    y_pred = best_rf.predict(test_X)
    # if(config.task_name == 'multi'):
        # _multi_metrics(config, test_y, y_pred)
    # else:
    _metrics(test_y, y_pred)
        # write_wrong_sample(config, test_y, y_pred, test_names)
        # write_black_sample(config, test_y, y_pred, test_names)


def get_pred(black_proba, white_proba):
    y_pred = []
    for (black_black, black_white), (white_black, white_white) in zip(black_proba, white_proba):
        if(black_black > black_white and white_black >= white_white): y_pred.append(0)
        elif(black_black > black_white and white_black < white_white): y_pred.append(0 if black_black >= white_white else 1)
        elif(black_black < black_white and white_black > white_white): y_pred.append(0 if black_black >= white_white else 1)
        elif(black_black <= black_white and white_black < white_white): y_pred.append(1)
        else: y_pred.append(0)
    return y_pred

def biknn(config, data_loaders):
    (train_data_black_embedding, train_data_white_embedding, test_data_black_embedding, test_data_white_embedding), labels = data_loaders
    test_labels = labels[train_cut_index:]
    train_data_embedding = train_data_black_embedding + train_data_white_embedding
    train_data_labels = [0] * len(train_data_black_embedding) + [1] * len(train_data_white_embedding)

    k_range = [ 2 ** i for i in range(1, 7)]
    for k in k_range:
        knn = KNeighborsClassifier(k)
        knn.fit(train_data_embedding, train_data_labels)
        proba_black = knn.predict_proba(test_data_black_embedding)
        proba_white = knn.predict_proba(test_data_white_embedding)
        y_pred = get_pred(proba_black, proba_white)
        _metrics(test_labels, y_pred)

def train_ml(config, data_loaders):
    if(config.model_name == 'knn'):
        knn(config, data_loaders)
    elif(config.model_name == 'dtree'):
        dtree(config, data_loaders)
    elif(config.model_name == 'rsvc'):
        rsvc(config, data_loaders)
    elif(config.model_name == 'lsvc'):
        lsvc(config, data_loaders)
    elif(config.model_name == 'psvc'):
        psvc(config, data_loaders)
    elif(config.model_name == 'ssvc'):
        ssvc(config, data_loaders)
    elif(config.model_name == 'lr'):
        lr(config, data_loaders)
    elif(config.model_name == 'rf'):
        rf(config, data_loaders)
    elif(config.model_name == 'gnb'):
        gnb(config, data_loaders)
    elif(config.model_name == 'gbdt'):
        gbdt(config, data_loaders)
    elif(config.model_name == 'biknn'):
        biknn(config, data_loaders)