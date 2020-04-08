import recordlinkage as rl
import streamlit as st
import pandas as pd 
import numpy as np 
import altair as alt

from recordlinkage.index import Block
from recordlinkage.base import BaseIndexAlgorithm
from recordlinkage.index import Block,SortedNeighbourhood
from recordlinkage.compare import Exact, String, Numeric, Geographic, Date
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import logging

import uuid

from functools import wraps
import time

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print ("@timefn:" + fn.__name__ + " took " + str(t2 - t1) + " seconds")
        return result
    return measure_time

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
@st.cache
def _blocking(df_a,blocks):
    logging.info("running  blocking ....")      
    df_a = df_a.copy()
    indexer = BlockUnion(block_on= blocks)
    candidate_pairs  = indexer.index(df_a)
    
    return candidate_pairs

@timefn
@st.cache
def _comparison(df_a,candidate_pairs,comparison):
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
                x='precision',
                y='recall'
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
