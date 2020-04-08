import sys
import streamlit as st
import recordlinkage as rl
import pandas as pd 
import numpy as np 

from os import path
import altair as alt
from recordlinkage.preprocessing import clean,phonetic
import cProfile
import base64
import logging

import matching

_compare_vartype = ['exact','string']
_compare_string_method = ['jarowinkler','jaro','levenshtein']
_phonetic_encoding = ['metaphone','soundex']
_classifiers = ['SimSum','ECM']

       
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

@timefn
@st.cache(allow_output_mutation=True)
def load_data(file_name):
    logging.info("loading data ....")      
    
    if file_name is not None :  
        df = pd.read_csv(file_name,low_memory= False)
        return df
    else:   
        st.error("File not found")
        return
    
st.cache(allow_output_mutation=True)
def load_data_gold_standard(filename,is_gold_standard):
    
    if is_gold_standard :
        if filename is not None :
            df_true = pd.read_csv(filename)
            return df_true
        else:
            st.error("File not found")
            return


@timefn
@st.cache(allow_output_mutation=True)
def set_data_index(df_a,index_field):
    
    logging.info("setting index....")      
    if (df_a is not None) :
        df_a = df_a.set_index(index_field)
    
    return df_a
    

def set_data_index_true_links(df_true_links,index_field_1,index_field_2,is_gold_standard):
   
    if is_gold_standard:
        if (df_true_links is not None): 
            df_true_links = df_true_links.set_index([index_field_1,index_field_2])
    
    if is_gold_standard:    
        return df_true_links
    else:
        return None 


@timefn
def show_ui_phonetic_encoding(df_a,index_name):
    logging.info("UI- show_ui_phonetic_encoding...")
    st.markdown("---")
    select_encoding = {}
    cols = df_a.columns.to_list()
    
    encoding_cols = list(cols)
  
    st.header("Step 3- Tell us the Pre-processing or Encoding you want on each field")
    for i,c in enumerate(encoding_cols):
        if st.checkbox(c,key="ppp_check"+c) :
            ph_selected = st.selectbox("phonetic encoding:",_phonetic_encoding,key='phonetic_'+c)
            select_encoding[c] = ph_selected
    
    return select_encoding
       
@timefn             
@st.cache(allow_output_mutation= True)
def run_phonetic_encoding(df_a,select_encoding):
    logging.info("run phonetic encoding ....")      
         
    for field,encoding in select_encoding.items():
        df_a[encoding+"_"+field]= phonetic(clean(df_a[field]),method=encoding)
    
    cols = df_a.columns.to_list()      
    
    return df_a, cols

@timefn
def show_ui_blocking( cols):
    logging.info("UI- show_ui_blocking...")
    st.markdown("---") 
    st.header('Step 4 -Now, please configure the blocking strategy you want ')
    number_blocks = st.number_input("Number of predicates:",1) + 1
    options = [st.empty() for i in range(1,number_blocks)]
    options_cols =[[]  for i in enumerate(options)]
    for i,opt in enumerate(options):
       options_cols[i] = st.multiselect('Select blocking fields',cols,cols[:2],key="options"+str(i))
    
    return options_cols


def show_ui_comparison(cols,_compare_vartype,_compare_string_method):
    logging.info("UI- show_ui_comparison...")
    st.markdown("---")
    st.header('Step 5 - Comparison')
    list_checkbox_fields = [st.empty for i in range(1,len(cols))]
    select_fields ={}
    comparison = []
    for i, col in enumerate(cols):
        item = {}
        if st.checkbox(cols[i]):
            vartype = st.selectbox("vartype",_compare_vartype,key="vartype_"+col)  
            if(vartype =='string'):
                method = st.selectbox('method',_compare_string_method,key="method"+col)
                if(st.checkbox('set threshold',key='threshold_check'+col)):
                    threshold = st.number_input('threshold',key='threshold_'+col)
                    item['threshold'] = threshold
                item['method'] = method
            item['vartype'] = vartype
            item['field'] = col
            item['code'] = col
            comparison.append(item)
            st.markdown("---")
        if len(comparison) == 0 :
            comparison.append({'vartype':'exact','field': cols[0],'code':cols[0]})
            comparison.append({'vartype':'exact','field': cols[1],'code':cols[1]})
    
    return comparison         
    
@timefn    
def show_ui_import_data():
    st.header("Step 1- Let's **begin** with the dataset you want to deduplicate :")
    uploaded_file  = st.file_uploader("Browse and select the file:",type = "csv")
    
    return uploaded_file
    



def show_ui_import_gold_standard(): 
    st.markdown("---")
    st.header("Step 2- Do you have a **gold standard** for this dataset ?")
    st.info("##### In order to asses the quality of the different matching strategies ground-thruth data , "
                    "known as gold standard are required")
    is_gold_standard = True if st.radio("Check Yes or No if you have a gold standard ?",("Yes","No"))  == "Yes" else False
        
    if(is_gold_standard) :
        uploaded_true = st.file_uploader("Browse and select the gold standard (Review instructions for the correct format of the gold standard):",type = "csv")
        
        if uploaded_true is not None :
            return uploaded_true,is_gold_standard
        else:
            return None,is_gold_standard
    else:
        return None, is_gold_standard



@timefn
def show_ui_set_index(cols):
    st.write("## Select the  record identifier in your dataset")
    # UI set index
    if cols is not None:
        index_name = st.selectbox("Record Identifier:",cols)
        return index_name
    else:
        return None
        

def show_ui_set_index_true_links(cols_true,is_gold_standard):
      
    if ( (is_gold_standard) & (cols_true is not None)):
        index_name_1 = st.selectbox("Record Identifier 1:",cols_true)
        index_name_2 = st.selectbox("Record Identifier 2:",cols_true)

        return index_name_1,index_name_2
    else:
        return None, None



def show_ui_classification(features):
    st.markdown("---")
    st.header('Step 6 -  Classification')
    option_classifiers = []
    threshold = 1
    for i , classifier in enumerate(_classifiers):
        if st.checkbox(classifier) :
            option_classifiers.append(classifier)
            if classifier == 'SimSum' :
                threshold = st.slider("Threshold :",1,len(features.columns)+1)
                
    return option_classifiers, threshold


def get_table_download_link(df,data_name):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download {data_name} as csv file</a>'
    
    return href
                                                              
@timefn                     
def show_ui_leaderboard(app_results,option_classifiers,is_gold_standard):
    logging.info("UI- show_ui_leaderboard...")
    st.markdown("---")   
     
    df_a = app_results['data'] 

    features = app_results['comparison_vector']
    index_name = app_results['index_name']
    if is_gold_standard :
        index_name_1 = app_results['index_name_1']
        index_name_2 = app_results['index_name_2']
        df_true_links = app_results['df_true_links']
 
    results_dict = {}
 
    for i , select_classifier in enumerate(option_classifiers):
        matches = app_results[select_classifier]['matches']
        decision_proba = app_results[select_classifier]['decision_proba']
        
        results_dict[select_classifier] = {}
        results_dict[select_classifier]['matches'] = matches
        m = matches.to_frame(index= False).columns.to_list()
        
        results_dict[select_classifier]['unique'] = matching.get_unique(df_a,matches,index_name,m[0],m[1])
        results_dict[select_classifier]['metrics']  = {}
        results_dict[select_classifier]['metrics']['nunique'] = results_dict[select_classifier]['unique']['nunique']    
        
        if is_gold_standard :
            results_dict[select_classifier]['unique'] = matching.get_unique(df_a,matches,index_name,index_name_1,index_name_2)
            results_dict[select_classifier]['metrics'] =  matching.metrics(df_true_links,matches,features)    
            results_dict[select_classifier]['matrix'] = rl.confusion_matrix(df_true_links,matches,len(features))
            results_dict[select_classifier]['roc'] = matching.show_roc_curve(df_true_links,decision_proba)
            results_dict[select_classifier]['pr'] = matching.show_precision_recall_curve(df_true_links,decision_proba)
                                                        
    for classifier, results in results_dict.items():
        st.markdown("## "+classifier)
        st.markdown("___")
        
        if  not is_gold_standard :
            unique_data = results['unique']['data_unique']
            nunique = results['metrics']['nunique']
            st.write("Number of unique records : "+ str(nunique))
            #st.write("Number of duplicate records : "+ str(len(matches)))
            st.write(unique_data)
            st.markdown(get_table_download_link(unique_data,""),unsafe_allow_html=True)
        else:    
            st.table(pd.DataFrame(results_dict[classifier]['metrics'],index=['metrics']) )          
            
            if st.checkbox("Show details",key="leader_details"+classifier):
                
                st.markdown("### Confusion matrix")
                matrix = pd.DataFrame(results['matrix'], columns=['Predicted Positives','Predicted Negatives'],index = ['True Positives','True Negatives'])
                st.table(matrix)
                
                st.markdown("### Visuals")
                st.altair_chart(results['roc'])
                st.altair_chart(results['pr'])
                if st.checkbox('Show confusion matrix data',key='show_data'+classifier):
                    selected_data = st.selectbox("select dataset",['matches',
                                                                'non-matches',
                                                                'true positives',
                                                                'false positives',
                                                                'true negatives',
                                                                'false negatives',
                                                                ],key='data_'+classifier)
                                        
                    nb_data = st.empty()
                    data = matching.get_confusion_matrix_data(selected_data,df_a,
                                                            df_true_links,
                                                                        results['matches'],
                                                                        features,
                                                                        index_name_1,
                                                                        index_name_2)
                    st.write(selected_data +": "+str(len(data))+" records pairs")
                    st.write(data)
                    st.markdown(get_table_download_link(data,selected_data),unsafe_allow_html=True)
                if st.checkbox("Show unique records",key="show_unique_"+classifier):
                    unique_data = results['unique']['data_unique']
                    st.write(unique_data[:1000])
                    st.markdown(get_table_download_link(unique_data,""),unsafe_allow_html=True)
 
@timefn    
def run_app():   
    
    logging.info("start running app....")
    st.title("Interactive Record Linkage Toolkit")
        
    # UI Import data
    uploaded_file = show_ui_import_data()    
    
    if uploaded_file is not None:       
        if st.checkbox('Import data') :      
            df_a = load_data(uploaded_file)    
            cols = df_a.columns.to_list()
                       
            index_name = show_ui_set_index(cols)
            df_a = set_data_index(df_a,index_name)
            
                # Show data to deduplicate
            st.markdown("### First 5 rows of imported data")
            st.write(df_a.head())
            
            # gold stardard
            uploaded_file_true_links,is_gold_standard = show_ui_import_gold_standard()
            
            
            cols_true = []
            index_name_1 = None
            index_name_2 = None
            df_true_links = None
            if is_gold_standard :
                if st.checkbox("Import gold standard"):
                    df_true_links  = load_data_gold_standard(uploaded_file_true_links,is_gold_standard)
                    if df_true_links is not None:
                        cols_true = df_true_links.columns.to_list()
                        # UI set Index
                        index_name_1,index_name_2 = show_ui_set_index_true_links(cols_true,is_gold_standard)      
        
                        # set index
                        df_true_links = set_data_index_true_links( df_true_links,index_name_1,index_name_2,is_gold_standard)
                        
                                     # Show data to deduplicate
                        st.markdown("### First 5 rows of gold standard")
                        st.write(df_true_links.index.to_frame(index=False).head())
                        
            
   
            
            # Set phonetic encoding
            selected_encoding = show_ui_phonetic_encoding(df_a,index_name)
            if st.checkbox("Run Pre-processing") :
                df_a, cols = run_phonetic_encoding(df_a,selected_encoding)
                
          
                # Blocking  
                options_cols = show_ui_blocking(cols)
                
                if st.checkbox("Run Indexing"):
                    candidate_pairs = matching._blocking(df_a,options_cols)
                    st.markdown('**'+str(len(candidate_pairs))+'** pairs')
                
                    # Comparison
                    comparison = show_ui_comparison(cols,_compare_vartype,_compare_string_method)
        
                    if st.checkbox("Run Comparison"):
                        features = matching._comparison(df_a,candidate_pairs,comparison)
                        
                        # UI Classification 
                        option_classifiers, threshold = show_ui_classification(features)               
            
                        if st.checkbox('Run Classification') : 
                            # classification
                            
                            app_results = {}
                            app_results['data'] = df_a
                            app_results['index_name'] = index_name
                            app_results['comparison_vector'] = features
                            
                            if is_gold_standard :
                                app_results['df_true_links'] = df_true_links
                                app_results['index_name_1'] = index_name_1
                                app_results['index_name_2'] = index_name_2          
                        
                            
                            # message when each one is complete
                            for i , select_classifier in enumerate(option_classifiers):
                                if select_classifier == 'SimSum' :
                                    matches,decision_proba = matching.simSum(features,threshold)
                                elif select_classifier == 'ECM' :
                                    matches,decision_proba  = matching.em_classifier(features)
                                else:
                                    pass
                                
                        
                                app_results[select_classifier] = {}   
                                app_results[select_classifier]['matches'] = matches
                                app_results[select_classifier]['decision_proba'] = decision_proba        
                            
                    
                            if st.checkbox("Show Leaderboard"):
                                st.markdown('## Leaderboard')
                                show_ui_leaderboard(app_results,option_classifiers,is_gold_standard)
        logging.info("end running app....")                              

def main():
    logging.basicConfig(level=logging.INFO)
    st.sidebar.title("Menu")       
    app_mode = st.sidebar.selectbox("Please select a page", ["Show Instructions",
                                                             "Run Deduplication"])
    app_results = {}
    if(app_mode == 'Show Instructions'):
        st.title("App Instructions")
    elif(app_mode == "Run Deduplication"):
        run_app()

                
if __name__ == "__main__":
    main()



