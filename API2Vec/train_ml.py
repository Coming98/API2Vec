from itertools import chain
import pickle
from statistics import mean
import time
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score, multilabel_confusion_matrix, plot_roc_curve, roc_curve, \
        auc
import matplotlib.pyplot as plt
import utils
import numpy as np
import os
import pandas as pd

def multi_single_analysis(config, test_y, test_names, pred_y):

    multi_test_y, multi_pred_y = [], []
    single_test_y, single_pred_y = [], []

    for i, name in enumerate(test_names):
        brief_name = utils.path2name(name)
        
        if(brief_name not in config.infos): 
            print("Wrong Name")
            exit(0)
        
        params = config.infos[brief_name]
        if(params['pid_count'] > 2):
            multi_test_y.append(test_y[i])
            multi_pred_y.append(pred_y[i])
        else:
            single_test_y.append(test_y[i])
            single_pred_y.append(pred_y[i])

    print("Multi Metrics")
    _metrics(multi_test_y, multi_pred_y)

    print("Single Metrics")
    _metrics(single_test_y, single_pred_y)

def multi_single_analysis_iter(config, test_y, test_names, pred_y):

    datas = {}

    for i, name in enumerate(test_names):
        brief_name = utils.path2name(name)
        
        if(brief_name not in config.infos): 
            print("Wrong Name")
            exit(0)
        
        pid_count = config.infos[brief_name]['pid_count'] - 1
        if(pid_count >= 9): pid_count = 9
        if(pid_count not in datas): datas[pid_count] = { 'test_y': [], 'pred_y': [] }
        datas[pid_count]['test_y'].append(test_y[i])
        datas[pid_count]['pred_y'].append(pred_y[i])

    # for key, params in datas.items():
    #     print(f"PID_COUNT = {key}")
    #     _metrics(params['test_y'], params['pred_y'])
    
    for pid_count in range(1, 9):
        print(f"PID_COUNT = {pid_count}")
        _metrics(datas[pid_count]['test_y'], datas[pid_count]['pred_y'])        

def print_roc_infos(model, X, y):
    y_pred_pro = model.predict_proba(X)[:, 1]
    fprs, tprs, thres = roc_curve(y, y_pred_pro, )
    # data = pd.DataFrame({'Y': y, 'Pred': y_pred_pro})
    # data.to_excel("temp2.xlsx", index=False)
    res_auc = auc(fprs, tprs)

    # with open('auc_datas.pkl', 'wb') as f:
    #     pickle.dump({'node2vecd': (fprs, tprs)}, f)

    # auc_datas = utils.pickle_load('auc_datas.pkl')
    # auc_datas['node2vecb'] = (fprs, tprs)
    # print(auc_datas.keys())
    # utils.pickle_dump(auc_datas, 'auc_datas.pkl')
    print("fprs\n", [f'{item:.6f}' for item in fprs])
    print("tprs\n", [f'{item:.6f}' for item in tprs])
    print("auc: ", res_auc)

    return

def _metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    # 精确率: 不漏报的能力 - 不将负样本标记为正样本的能力
    # precision = precision_score(y_true, y_pred)
    # TNR = tn / (fp + tn)
    # TPR = tp / (tp + fn)
    # 召回率: 不误判的能力 - 不将正样本标记为负样本的能力
    # recall = recall_score(y_true, y_pred)
    # 
    # F1-score: 模型的稳健能力
    # f1 = f1_score(y_true, y_pred)
    # black's label = 1, white's label = 0
    # tn, tp = tp, tn
    # fn, fp = fp, fn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    res = f'| tp={tp} | fn={fn} | tn={tn} | fp={fp} |\n'
    res += f'| precision={precision:.2%} | recall={recall:.2%} | f1={f1:.2%} | acc={acc:.2%} |\n'
    print(res)
    with open('result.temp.txt', 'a', encoding='utf-8') as f:
        f.write("NEW==========================================" + '\n')
        f.write(res + '\n')

def _model(config, data_loaders, model, datanames, train_cut_index=None):
    X, y = data_loaders
    train_X, test_X, train_y, test_y, train_names, test_names = _train_test_split(config, X, y, datanames, test_size=0.3, random_state=3, train_cut_index=train_cut_index)

    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    _metrics(test_y, y_pred)

def _train_test_split(config, X, y, names, test_size=0.3, random_state=3, train_cut_index=None):

    if(config.task_name == 'attack'):
        train_index = []
        test_index = []
        test_attack_index = []
        for i, name in enumerate(names):
            name = utils.path2name(name)
            if(name[-2:] == '_A'): test_attack_index.append(i)
            elif(name in config.against_sample_names): 
                test_index.append(i)
                # 白样本不需要对抗, 但是在检测时需要存在
                if(y[i] == 1):
                    test_attack_index.append(i)
            else: train_index.append(i)

        train_X = [X[i] for i in train_index]
        train_y = [y[i] for i in train_index]
        train_names = [names[i] for i in train_index]
        test_X = [X[i] for i in test_index]
        test_y = [y[i] for i in test_index]
        test_names = [names[i] for i in test_index]
        test_attack_X = [X[i] for i in test_attack_index]
        test_attack_y = [y[i] for i in test_attack_index]
        test_attack_names = [names[i] for i in test_attack_index]

        return train_X, (test_X, test_attack_X), train_y, (test_y, test_attack_y), train_names, (test_names, test_attack_names)
    
    if(config.task_name == 'mvs'):
        train_index = []
        test_index = []
        for i, name in enumerate(names):
            name = utils.path2name(name)
            if(name in config.test_names): 
                test_index.append(i)
            else: 
                train_index.append(i)

        train_X = [X[i] for i in train_index]
        train_y = [y[i] for i in train_index]
        train_names = [names[i] for i in train_index]
        test_X = [X[i] for i in test_index]
        test_y = [y[i] for i in test_index]
        test_names = [names[i] for i in test_index]

        print("test len: ", len(test_y))

        return train_X, test_X, train_y, test_y, train_names, test_names

    if(config.task_name == 'target'):
        #  X, y, names
        # TEST names
        target_types = config.params
        target_names = list(chain.from_iterable([config.type_names[type] for type in target_types if type in config.type_names]))
        
        X_white, X_black_train, X_black_test = [], [], []
        X_white_names, X_black_train_names, X_black_test_names = [], [], []
        for i, name in enumerate(names):
            brief_name = name.split('\\')[-1].split('.')[0]
            if(brief_name in target_names):
                X_black_test.append(X[i])
                X_black_test_names.append(name)
            else:
                if(y[i] == 0): # black
                    X_black_train.append(X[i])
                    X_black_train_names.append(name)
                else:
                    X_white.append(X[i])
                    X_white_names.append(name)
        test_ratio = len(X_black_test) / (len(X_black_test) + len(X_black_train))
        X_white_train, X_white_test, X_white_train_names, X_white_test_names = [], [], [], []

        shuffled_indexes = np.random.permutation(len(X_white))
        test_size = int(len(X_white) * test_ratio)
        train_index = shuffled_indexes[test_size:]
        test_index = shuffled_indexes[:test_size]

        X_white_train = [X_white[i] for i in train_index]
        X_white_train_names = [X_white_names[i] for i in train_index]
        X_white_test = [X_white[i] for i in test_index]
        X_white_test_names = [X_white_names[i] for i in test_index]

        train_X = X_white_train + X_black_train
        test_X = X_white_test + X_black_test
        train_y = [1, ] * len(X_white_train) + [0, ] * len(X_black_train)
        test_y = [1, ] * len(X_white_test) + [0, ] * len(X_black_test)
        train_names = X_white_train_names + X_black_train_names
        test_names = X_white_test_names + X_black_test_names

        print("Train total:", len(train_X))
        print("Train black:", len(X_black_train))
        print("Train white:", len(X_white_train))
        print("Test total:", len(test_X))
        print("Test black:", len(X_black_test))
        print("Test white:", len(X_white_test))
        
        return train_X, test_X, train_y, test_y, train_names, test_names

    if train_cut_index is None:

        if random_state:
            np.random.seed(random_state)

        shuffled_indexes = np.random.permutation(len(X))
        test_size = int(len(X) * test_size)
        train_index = shuffled_indexes[test_size:]
        test_index = shuffled_indexes[:test_size]

        train_X = [X[i] for i in train_index]
        train_y = [y[i] for i in train_index]
        train_names = [names[i] for i in train_index]
        test_X = [X[i] for i in test_index]
        test_y = [y[i] for i in test_index]
        test_names = [names[i] for i in test_index]
    else:
        train_X = X[:train_cut_index]
        train_y = y[:train_cut_index]
        train_names = names[:train_cut_index]
        test_X = X[train_cut_index:]
        test_y = y[train_cut_index:]
        test_names = names[train_cut_index:]

    # 根据 name 筛选测试集 1000 个多进程, 1000 个单进程
    # multi_test_X, single_test_X = [], []
    # multi_test_y, single_test_y = [], []
    # multi_test_names, single_test_names = [], []

    # for i, name in enumerate(test_names):
    #     name = name.split('\\')[-1]
    #     if(config.infos[name]['pid_count'] > 2):
    #         multi_test_X.append(test_X[i])
    #         multi_test_y.append(test_y[i])
    #         multi_test_names.append(test_names[i])
    #     else:
    #         single_test_X.append(test_X[i])
    #         single_test_y.append(test_y[i])
    #         single_test_names.append(test_names[i])

    # print(len(test_names))
    # print(len(multi_test_X))
    # print(len(single_test_X))
    # # exit(0)
    # C = [1000, 2000, 4000][0]
    # flag = [0, 1][1]
    # if(flag):
    #     print(f"Multi Test: {C}")
    #     test_X = multi_test_X[:C]
    #     test_y = multi_test_y[:C]
    #     test_names = multi_test_names[:C]
    # else:
    #     print(f"Single Test: {C}")
    #     test_X = single_test_X[:C]
    #     test_y = single_test_y[:C]
    #     test_names = single_test_names[:C]


    # against_sample_names = []
    # for i, name in enumerate(test_names):
    #     against_sample_names.append(utils.path2name(name))

    return train_X, test_X, train_y, test_y, train_names, test_names

def write_wrong_sample(config, y_true, y_pred, names):
    target_indexes = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    filtered_names = [ names[i] for i in target_indexes ]
    black_count = len(list(filter(lambda name: 'black' in name, filtered_names)))
    white_count = len(filtered_names) - black_count
    save_path = os.path.join(config.wrong_sample_output_dir, f'{config.model_name}-{config.data_name}-{config.pid_flag}[{config.api_count_min}-{config.sequence_length_min}]' + (f'[{config.rw_k}-{config.rw_step}]' if config.pid_flag == 'pid' else '') + f'-{config.epochs}-{config.vector_size}-{config.model_type}-{config.data_type}.html')
    save_path = './result.targetnames.html'
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(f'<h3>Black Count: {black_count}; White Count: {white_count}</h3><br>')
        for name in filtered_names:
            f.write(f'<a href="file:///{name}">{name}</a><br>')

def write_black_sample(CONFIG, y_true, y_pred, names):
    return
    save_dir = './outputs/black_family'
    save_path = os.path.join(save_dir, CONFIG.black_family_name)
    pred_black_names = dict({
        "true": [],
        "false": []
    })
    for i in range(len(y_true)):
        if(y_true[i] == 1): continue # 针对恶意样本类别 忽略白羊本
        if(y_pred[i] == y_true[i]):
            pred_black_names['true'].append(names[i])
        else:
            pred_black_names['false'].append(names[i])
    with open(save_path, 'wb') as f:
        pickle.dump(pred_black_names, f)

def gnb(config, data_loaders, datanames):
    model = GaussianNB()
    _model(config, data_loaders, model, datanames)
def mnb(config, data_loaders, datanames):
    model = MultinomialNB()
    _model(config, data_loaders, model, datanames)
def gbdt(config, data_loaders, datanames):
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
def rsvc(config, data_loaders, datanames):
    model = SVC()
    _model(config, data_loaders, model, datanames)
def lsvc(config, data_loaders, datanames):
    model = LinearSVC()
    _model(config, data_loaders, model, datanames)
def psvc(config, data_loaders, datanames):
    model = SVC(kernel='poly')
    _model(config, data_loaders, model, datanames)
def ssvc(config, data_loaders, datanames):
    model = SVC(kernel='sigmoid')
    _model(config, data_loaders, model, datanames)
def lr(config, data_loaders, datanames):
    model = LogisticRegression(max_iter=9000)
    _model(config, data_loaders, model, datanames)

def dtree(config, data_loaders, datanames, train_cut_index):
    X, y = data_loaders
    train_X, test_X, train_y, test_y, train_names, test_names = _train_test_split(config, X, y, datanames, test_size=0.3, random_state=3, train_cut_index=train_cut_index)

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
    write_black_sample(config, test_y, y_pred, test_names)

def knn(config, data_loaders, datanames, train_cut_index):

    X, y = data_loaders
    train_X, test_X, train_y, test_y, train_names, test_names = _train_test_split(config, X, y, datanames, test_size=0.3, random_state=3, train_cut_index=train_cut_index)
    print("Train: ", len(train_X))
    print("Test: ", len(test_X))

    # k_range = [ 2 ** i for i in range(1, 7)]
    k_range = [ i for i in range(2, 10)]
    scores = []

    for k in k_range:
        knn = KNeighborsClassifier(k)
        cv_scores = cross_val_score(knn, train_X, train_y, cv=10, scoring='accuracy')
        scores.append(cv_scores.mean())

        print(f'k = {k}, mean_score = {cv_scores.mean()}\nscores = {cv_scores}')

    # utils.show_plot(k_range, scores, 'k', 'accuracy', save=False)

    best_mean_scores_index = scores.index(max(scores))
    best_k = k_range[best_mean_scores_index]
    print(f"Mean scores = {scores}")
    print(f"best mean score = {scores[best_mean_scores_index]}")
    print(f"best k = {best_k}")

    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(train_X, train_y)

    if(config.task_name == 'attack'):
        test_X, test_attack_X = test_X
        test_y, test_attack_y = test_y
        test_names, test_attack_names = test_names
        # normal test
        print("NORMAL TEST")
        y_pred = best_knn.predict(test_X)
        _metrics(test_y, y_pred)
        print("Attack TEST")
        y_pred = best_knn.predict(test_attack_X)
        _metrics(test_attack_y, y_pred)
        return
        
    start_time = time.time()
    y_pred = best_knn.predict(test_X)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Time - Predict(count = {len(y_pred)}) = {total_time}")
    
    _metrics(test_y, y_pred)
    print(best_knn.score(test_X, test_y))
    if(config.task_name == 'normal'):
        write_wrong_sample(config, test_y, y_pred, test_names)
        # write_black_sample(config, test_y, y_pred, test_names)

        # print_roc_infos(best_knn, test_X, test_y)

        # multi_single_analysis(config, test_y, test_names, y_pred)
        # multi_single_analysis_iter(config, test_y, test_names, y_pred)

def rf(config, data_loaders, datanames, train_cut_index):
    X, y = data_loaders
    
    train_X, test_X, train_y, test_y, train_names, test_names = _train_test_split(config, X, y, datanames, test_size=0.3, random_state=3, train_cut_index=train_cut_index)

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

    _metrics(test_y, y_pred)
    write_wrong_sample(config, test_y, y_pred, test_names)
    write_black_sample(config, test_y, y_pred, test_names)

def train_ml(config, data_loaders, datanames, train_cut_index):
    if(config.model_name == 'knn'):
        knn(config, data_loaders, datanames, train_cut_index)
    elif(config.model_name == 'dtree'):
        dtree(config, data_loaders, datanames, train_cut_index)
    elif(config.model_name == 'rsvc'):
        rsvc(config, data_loaders, datanames)
    elif(config.model_name == 'lsvc'):
        lsvc(config, data_loaders, datanames)
    elif(config.model_name == 'psvc'):
        psvc(config, data_loaders, datanames)
    elif(config.model_name == 'ssvc'):
        ssvc(config, data_loaders, datanames)
    elif(config.model_name == 'lr'):
        lr(config, data_loaders, datanames)
    elif(config.model_name == 'rf'):
        rf(config, data_loaders, datanames, train_cut_index)
    elif(config.model_name == 'gnb'):
        gnb(config, data_loaders, datanames)
    elif(config.model_name == 'gbdt'):
        gbdt(config, data_loaders, datanames)