import recordlinkage as rl
import streamlit as st
import pandas as pd 
import numpy as np 
import altair as alt

from recordlinkage.preprocessing import clean,phonetic
from recordlinkage.index import Block
from recordlinkage.base import BaseIndexAlgorithm
from recordlinkage.index import Block,SortedNeighbourhood
from recordlinkage.compare import Exact, String, Numeric, Geographic, Date
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from sklearn.model_selection import KFold,StratifiedKFold
from metaphone import doublemetaphone

import logging
import uuid
import copy
from utils import timefn
from functools import reduce



class BlockUnion(BaseIndexAlgorithm):
    def __init__(self,block_on = None, **kwargs):
        
        super(BlockUnion,self).__init__(**kwargs)
        self.block_on = block_on         
         
    def _link_index(self, df_a, df_b):
        indexer = rl.Index()
          
        for blocking_keys in self.block_on:
            indexer.add(Block(blocking_keys))
        
        return  indexer.index(df_a,df_b)
     
    def _dedup_index(self,df_a):
         
        indexer = rl.Index()
         
        for blocking_keys in self.block_on:
            indexer.add(Block(blocking_keys))
          
        return  indexer.index(df_a)


class Comparator():
    
    def __init__(self,compare_on = None, **kwargs):
        super(Comparator,self).__init__(**kwargs)
        self.compare_on = compare_on
        self.comparator = rl.Compare()
            
    def compute_compare(self,pairs,df_a,df_b=None):
        for comparison in self.compare_on:
            vartype,field,method, threshold, label = self._unpack_dict(**comparison)
            if vartype == 'string':
                self.comparator.add(String(field,field,method,threshold,0,label))
            if vartype == 'exact':
                self.comparator.add(Exact(field,field,1,0,0,label))
                
        return  self.comparator.compute(pairs,df_a,df_b)
    
    def _unpack_dict(self,vartype = None,field= None,method = None,threshold = None, code= None):
        return vartype,field,method, threshold, code


@timefn             
@st.cache(allow_output_mutation= True)
def run_phonetic_encoding(df_a,select_encoding):
    logging.info("run phonetic encoding ....")      
    df_a_processed = df_a.copy()
    
    #FIXME Errors when selecting non string columns like soc_sec_id
    #TODO Include double metaphone in Python Toolkit
    for field,encoding in select_encoding.items():
        if (encoding =='double_metaphone'):
             df_a_processed[encoding+"_"+field] = df_a[field].apply(lambda x: doublemetaphone(str(x))[0] if(np.all(pd.notnull(x))) else x)
        else:
         df_a_processed[encoding+"_"+field]= phonetic(clean(df_a[field]),method=encoding)
    
    cols = df_a_processed.columns.to_list()      

    return df_a_processed, cols


@timefn
@st.cache
def run_blocking(df_a, blocks,blocking ="Standard"):
    
    logging.info("running  blocking ....")      
    df_a = df_a.copy()

    if (blocking == 'Standard'):
        indexer = BlockUnion(block_on= blocks)
        candidate_pairs  = indexer.index(df_a)
    if (blocking == "Full Indexing"):
        indexer = rl.Index()
        indexer.full()
        candidate_pairs  = indexer.index(df_a)
    elif(blocking == 'SortedNeighbourhood'):
        key = 'given_name'
        indexer = SortedNeighbourhood(key,window=3)
        candidate_pairs  = indexer.index(df_a)
    else:
        indexer = BlockUnion(block_on= blocks)
        candidate_pairs  = indexer.index(df_a)
    
    return candidate_pairs


@timefn
@st.cache
def run_comparison(df_a,candidate_pairs,comparison):
    logging.info("running comparison....")      
    df_a = df_a.copy()
    compare_cl = Comparator(compare_on = comparison)
    features = compare_cl.compute_compare(candidate_pairs, df_a) 
     
    return features    


@st.cache
def simSum(features, threshold):
    df_f = features.copy()
    df_f['score'] = features.sum(axis=1)
    #threshold or score based classification
    matches = df_f[df_f['score'] >= threshold]
    
    return matches.index, df_f['score']

@st.cache
def  exact_matching_classifier(candidate_pairs):
    """     Exact deterministic matching      """
    return candidate_pairs


@st.cache
def em_classifier(features):
    ecm  = rl.ECMClassifier(binarize=0.85)
    matches  = ecm.fit_predict(features)
    
    df_ecm_prob = pd.DataFrame(ecm.prob(features))
    df_ecm_prob.columns = ['score']
    return matches,df_ecm_prob


@st.cache
def kmeans_classifier(features):
    kmeans = rl.KMeansClassifier()
    matches = kmeans.fit_predict(features)
    
    return matches

##FIXME Fix ML classifiers to accept train/test split

@st.cache
def logreg_classifier(features,links_true,train_size = 0.2,cv=None):
    logreg = rl.LogisticRegressionClassifier()
    
    if cv is None:
        golden_match_index = features.index & links_true.index
        train_index = int(len(features)*train_size)
        #train model
        logreg.fit(features[0:train_index], golden_match_index)

        # Predict the match status for all record pairs
        matches = logreg.predict(features)
        
        df_logreg_prob = pd.DataFrame(logreg.prob(features))
        df_logreg_prob.columns = ['score']
    else :
        df_results = cross_val_predict(logreg,features,links_true,cv,method='predict')
        matches = df_results.index
        df_logreg_prob = cross_val_predict(logreg,features,links_true,cv,method='predict_proba')
    
    return matches, df_logreg_prob

@st.cache
def nb_classifier(features,links_true,train_size = 0.2,cv=None):
    nb = rl.NaiveBayesClassifier(binarize=0.3)
    print(features)
    print(len(features))
    ##FIXME train_size check should be greater than 0  less than 1
    if cv is None:
        golden_match_index = features.index & links_true.index
        train_index = int(len(features)*train_size)
        print(train_index)
        #train model
        nb.fit(features[0:train_index], golden_match_index)

        # Predict the match status for all record pairs
        matches = nb.predict(features)
        
        df_nb = pd.DataFrame(nb.prob(features))
        df_nb.columns = ['score']
    else :
        df_results = cross_val_predict(nb,features,links_true,cv,method='predict')
        matches = df_results.index
        df_nb = cross_val_predict(nb,features,links_true,cv,method='predict_proba')
    
    return matches, df_nb


@st.cache
def svm_classifier(features,links_true,train_size = 0.2,cv=None):
    svm = rl.SVMClassifier()
    
    ##FIXME train_size check should be greater than 0  less than 1
    if cv is None:
        golden_match_index = features.index & links_true.index
        train_index = int(len(features)*train_size)
        #train model
        svm.fit(features[0:train_index], golden_match_index)

        # Predict the match status for all record pairs
        matches = svm.predict(features)
        
        df_svm = pd.DataFrame(svm.kernel.decision_function(features))
        df_svm.columns = ['score']
    else :
        df_results = cross_val_predict(svm,features,links_true,cv,method='predict')
        matches = df_results.index
        df_svm = cross_val_predict(svm,features,links_true,cv,method='decision_function')
    
    return matches, df_svm


@st.cache
def weighted_average_classifier(threshold,comparison_vectors,weight_factor):
    """  Weighted average matching  """
    
    
    ##FIXME we need to check if field match weight factor
    
    df_score = comparison_vectors.copy()
    weighted_list =[]
    factor_sum = 0        
    for field,wf in weight_factor.items():
        weighted_list.append(df_score[field]*int(wf))
        factor_sum += wf
    weighted_sum = reduce(lambda x, y: x.add(y, fill_value=0), weighted_list)
    df_score['score'] = weighted_sum/factor_sum
    
    matches = df_score[df_score['score'] >=threshold]
    
    
    return matches.index , matches['score']         
 
 
@st.cache
def cross_val_predict(classifier,comparison_vector,link_true,cv = 5 , method ='predict'):
        skfolds = StratifiedKFold(n_splits = cv)
        
        y = pd.Series(0, index=comparison_vector.index)
        y.loc[link_true.index & comparison_vector.index] = 1
        
        X_train = comparison_vector.values
        y_train = y.values
        
        results = pd.DataFrame()
        for train_index, test_index in skfolds.split(X_train,y_train):
            #clone_clf = clone(classifier)
            classifier_copy = copy.deepcopy(classifier)
            X_train_folds = comparison_vector.iloc[train_index]  #X_train[train_index]
            X_test_folds  = comparison_vector.iloc[test_index]  #X_train[test_index]
            y_train_folds = X_train_folds.index &  link_true.index #y_train[train_index]
            y_test_folds = X_test_folds.index & link_true.index

            # Train the classifier
            #print(y_train_folds.shape)
            classifier_copy.fit(X_train_folds, y_train_folds)

            # predict matches for the test
            #print(X_test_folds)
            
            if(method == 'predict'):
                y_pred = classifier_copy.predict(X_test_folds)
                results = pd.concat([results,y_pred.to_frame()])
            elif(method == 'predict_proba'):
                predict_proba = pd.DataFrame(classifier_copy.prob(X_test_folds))
                predict_proba.columns = ['score']
                results = pd.concat([results,predict_proba])
            elif(method == 'decision_function'):
                decision_match = classifier_copy.kernel.decision_function(X_test_folds.values)
                decision = pd.Series(decision_match,index = X_test_folds.index)
                df_decision = pd.DataFrame(decision)
                results = pd.concat([results,df_decision])

        return results 
   
    
@timefn
def metrics(links_true,links_pred,pairs):
    if len(links_pred) > 0 :
            
        # precision
        precision  = rl.precision(links_true, links_pred)

         #recall
        recall  = rl.recall(links_true, links_pred)

        # The F-score for this classification is
        fscore = rl.fscore(links_true,links_pred)
        
            
        return {'pairs':len(pairs),'#duplicates':len(links_pred),'precision':precision, 'recall':recall,'fscore':fscore}
    else :
        return {'pairs':0,'#duplicates':0,'precision':0, 'recall':0,'fscore':0}

@timefn
def run_metrics(app_results,option_classifiers,is_gold_standard):
    logging.info("Running metrics..")
    
    df_a = app_results['data'] 

    features = app_results['comparison_vector']
    index_name = app_results['index_name']
    if is_gold_standard :
        index_name_1 = app_results['index_name_1']
        index_name_2 = app_results['index_name_2']
        df_true_links = app_results['df_true_links']
 
    results_dict = {}
    
    # For each classifier calculate metrics in results_dict
    for i , select_classifier in enumerate(option_classifiers):
        matches = app_results[select_classifier]['matches']
        decision_proba = app_results[select_classifier]['decision_proba']
        
        results_dict[select_classifier] = {}
        results_dict[select_classifier]['matches'] = matches
        m = matches.to_frame(index= False).columns.to_list()
        
        results_dict[select_classifier]['unique'] = get_unique(df_a,matches,index_name,m[0],m[1])
        results_dict[select_classifier]['metrics']  = {}
        results_dict[select_classifier]['metrics']['nunique'] = results_dict[select_classifier]['unique']['nunique']    
        
        ##FIXME Can we separate metrics calculation from UI render
        if is_gold_standard :
            results_dict[select_classifier]['unique'] = get_unique(df_a,matches,index_name,index_name_1,index_name_2)
            results_dict[select_classifier]['metrics'] =  metrics(df_true_links,matches,features)    
            results_dict[select_classifier]['matrix'] = rl.confusion_matrix(df_true_links,matches,len(features))
            results_dict[select_classifier]['roc'] = show_roc_curve(df_true_links,decision_proba)
            results_dict[select_classifier]['pr'] = show_precision_recall_curve(df_true_links,decision_proba)
            
    return results_dict
  
  
def show_roc_curve(df_true_links,features):
    df_features_score = features.copy()
    #st.text(df_features_score.columns)
    
    y_scores = df_features_score.values
    Y = pd.Series(0, index= features.index)
    Y.loc[df_true_links.index & df_features_score.index] = 1
    fpr, tpr, thresholds = roc_curve(Y, y_scores)
    data = pd.DataFrame({'fpr':fpr,'tpr':tpr})
    
    c = alt.Chart(data).mark_line().encode(
                x='fpr',
                y='tpr'
            ).properties(
                     title = 'ROC Curve'
                ).interactive()
   
    return c
  
                
def show_precision_recall_curve(df_true_links,features):
    df_features_score = features.copy()
    
    y_scores = df_features_score.values
    Y = pd.Series(0, index= features.index)
    Y.loc[df_true_links.index & df_features_score.index] = 1
    
    precision, recall, thresholds  = precision_recall_curve(Y,y_scores)
    data = pd.DataFrame({'precision':precision,'recall':recall})
    
    c = alt.Chart(data).mark_line().encode(
                x='recall',
                y='precision'
            ).properties(
                title='Precision-Recall'
                ).interactive()
   
    return c   


@st.cache
def get_matches(matches):
    return matches

@st.cache
def get_true_positves(df_true_links,matches):
    return df_true_links.index & matches

@st.cache
def get_false_positives(df_true_links,matches):
    return matches.difference(df_true_links.index)

@st.cache
def get_false_negatives(df_true_links,matches):
    return df_true_links.index.difference(matches)

@st.cache
def get_true_negatives(df_true_links,matches,features):
    return features.index.difference(get_true_matches(df_true_links,matches,))

@st.cache
def get_non_matches(df_true_links,matches,features):
    # should get this whitout confusion matrix
    return get_false_positives(df_true_links,matches) | get_true_negatives(df_true_links,matches,features)

@st.cache
def get_true_matches(df_true_links,matches):
    return get_true_positves(df_true_links,matches) | get_false_negatives(df_true_links,matches)

@st.cache
def show_confusion_matrix_data(df_a,index_data,link_true_id_1,link_true_id_2):
    df_ = index_data.to_frame(index=False)
    return df_.join(df_a,on=[link_true_id_1]).join(df_a,on=[link_true_id_2], lsuffix='_1', rsuffix='_2')

@st.cache
def get_confusion_matrix_data(filter,df_a,df_true_links,matches,features,index_name_1,index_name_2):
    logging.info("getting confusion matrix data....")      
    switcher = {
        'true positives' : get_true_positves(df_true_links,matches),
        'true negatives' : get_true_negatives(df_true_links,matches,features),
        'false positives' : get_false_positives(df_true_links,matches),
        'false negatives' : get_false_negatives(df_true_links,matches),
        'matches' : get_matches(matches),
        'non-matches': get_non_matches(df_true_links,matches,features)
    }
    
    idx = switcher.get(filter)
    #idx = func()
    return show_confusion_matrix_data(df_a,idx,index_name_1,index_name_2)

        
class DisjointSet(object):

    def __init__(self):
        self.leader = {} # maps a member to the group's leader
        self.group = {} # maps a group leader to the group (which is a set)

    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])


@st.cache
def get_unique(df_a,df_true_links,index_field,index_field_1,index_field_2):
    logging.info("getting unique patient......")
    ds = DisjointSet()
    df_true_links_frame = pd.DataFrame()
    if(isinstance(df_true_links,pd.MultiIndex)):
        df_true_links_frame = df_true_links.to_frame(index=False)
        df_true_links_frame.columns=[index_field_1,index_field_2]
    else:
        df_true_links_frame = df_true_links.copy()
        df_true_links_frame = df_true_links.reset_index()
        
        # check index ??
    
    for index, row in df_true_links_frame.iterrows():
        ds.add(row[index_field_1],row[index_field_2])
    link_uid = []
    for el, item in ds.group.items():
        id = uuid.uuid4() 
        for val in item:
            link_uid.append((str(id),val))
    df_link_uid = pd.MultiIndex.from_tuples(link_uid,names=('uuid', index_field)).to_frame(index=False).set_index(index_field)
    df_a_new = df_a.merge(df_link_uid,how='left',left_on=index_field, right_on=index_field)
    df_a_new.loc[df_a_new['uuid'].isnull(),'uuid'] = df_a_new[df_a_new['uuid'].isnull()]['uuid'].apply(lambda x:str(uuid.uuid4()))
    
    return {'data_unique':df_a_new, 'nunique': df_a_new['uuid'].nunique()}
