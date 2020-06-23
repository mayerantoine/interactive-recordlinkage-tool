""" This module includes all streamlit function to display interactive widgets
"""

import logging
import streamlit as st
import base64
from matching import get_confusion_matrix_data
import pandas as pd

_phonetic_encoding = ['metaphone','double_metaphone','soundex','nysiis','match_rating']
_blocking = ['Full Indexing','Standard','SortedNeighbourhood']


def show_ui_import_data():
    st.header("Step 1- Let's **begin** with the dataset you want to deduplicate :")
    uploaded_file  = st.file_uploader("Browse and select the file (only accept CSV file):",type = "csv")
    
    return uploaded_file


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

def show_ui_import_gold_standard(): 
    st.markdown("---")
    st.header("Step 2- Do you have a **gold standard** for this dataset ?")
    st.info("""
            In order to asses the quality of the different matching strategies ground-thruth data , 
                    known as gold standard or labled data are required
            """)
    is_gold_standard = False if st.radio("Check Yes or No if you have a gold standard ?",("No","Yes"))  == "No" else True
        
    if(is_gold_standard) :
        uploaded_true = st.file_uploader("Browse and select the gold standard (Review instructions for the correct format of the gold standard):",type = "csv")
        
        if uploaded_true is not None :
            return uploaded_true,is_gold_standard
        else:
            return None,is_gold_standard
    else:
        return None, is_gold_standard 

def show_ui_phonetic_encoding(df_a,index_name):
    st.markdown("---")
    select_encoding = {}
    cols = df_a.columns.to_list()
    
    encoding_cols = list(cols)
  
    st.header("Step 3- Tell us the Pre-processing or Encoding you want on each field")
    st.info(""" 
             The main task of data cleaning and standardization is the conversion of the raw input data into well defined, consistent forms, as well as the resolution of inconsistencies 
             in the way information is represented and encoded.(Christen 2012)
             
             """)
    for i,c in enumerate(encoding_cols):
        if st.checkbox(c,key="ppp_check"+c) :
            ph_selected = st.selectbox("phonetic encoding:",_phonetic_encoding,key='phonetic_'+c)
            select_encoding[c] = ph_selected
    
    return select_encoding


def show_ui_blocking(cols):
    st.markdown("---") 
    st.header('Step 4 -Now, please configure the **blocking strategy** you want ')
    st.info(""" 
            The step of the process called blocking or indexing try to reduce the number of records we need to compare.
             The idea is instead of comparing all records of the dataset between themselves we want 
            to compare only the records that are most likely to be matched.
            """)
    blocking_selected = st.selectbox("Select blocking algorithm:",_blocking,key="_blocking_alg")
    
    #FIXME Catch error when blokcing fields empty
    if (blocking_selected == "Standard"):
        number_blocks = st.number_input("Number of predicates:",1) + 1
        options = [st.empty() for i in range(1,number_blocks)]
        options_cols =[[]  for i in enumerate(options)]
        for i,opt in enumerate(options):
            options_cols[i] = st.multiselect('Select blocking fields',cols,cols[:2],key="options"+str(i))
    else:
        options_cols = []
            
    return options_cols,blocking_selected


def show_ui_comparison(cols,_compare_vartype,_compare_string_method):
    st.markdown("---")
    st.header('Step 5 - Next, please configure the **comparison strategy** you want to apply')
    st.info(""" 
            In this step you select the similarity algorihtm and calculate
             the similarity score between records pairs to create a comparison vectors 
            """)
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
 




def show_ui_classification(features,df_a,list_classifiers):
    st.markdown("---")
    st.header('Step 6 -  Classification')
    st.info(""" 
            Based on comparison vector, this step uses a classification algorithm to
            classify candidate records pairs in: matches and non-matches.
            """)
    option_classifiers = []
    threshold = 1
    threshold_wavg = 1
    select_wf= {}
    for i , classifier in enumerate(list_classifiers):
        if st.checkbox(classifier) :
            option_classifiers.append(classifier)
            if classifier == 'SimSum' :
                threshold = st.slider("Threshold :",1,len(features.columns)+1)
            elif classifier == "Weighted Average":
                threshold_wavg = st.number_input("Threshold",min_value=0.1,max_value=0.999,key="threshold_wf")
                cols = features.columns.to_list()
      
                st.write("Select attributes weights:")
                for i,c in enumerate(cols):
                    if st.checkbox(c,key="wf_check_"+c+str(i)) :
                        wf_selected = st.number_input("weight :",value =1,key="wf_"+c)
                        select_wf[c] = wf_selected
                    else:
                        select_wf[c] = 1
    return option_classifiers, threshold,threshold_wavg,select_wf


def get_table_download_link(df,data_name):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download {data_name} as csv file</a>'
    
    return href


def show_ui_evaluation():
    st.markdown("---")
    st.header('Step 7 -  Evaluation')
    st.info(""" 
            Comparing match results with the known ground truth or gold standard 
            to measure the performance of the matching process.
            """)         


def show_ui_leaderboard(app_results,results_dict,option_classifiers,is_gold_standard):
    st.markdown('## Leaderboard') 
    df_a = app_results['data'] 

    features = app_results['comparison_vector']
    index_name = app_results['index_name']
    if is_gold_standard :
        index_name_1 = app_results['index_name_1']
        index_name_2 = app_results['index_name_2']
        df_true_links = app_results['df_true_links']
    
    # For each classifier display results    
    df_leaderboard = pd.DataFrame()                                             
    for classifier, results in results_dict.items():  
  
        if  not is_gold_standard :
            st.markdown("### "+classifier)    
            unique_data = results['unique']['data_unique']
            nunique = results['metrics']['nunique']
            st.write("Number of unique records : "+ str(nunique))
            st.write(unique_data)
            st.markdown(get_table_download_link(unique_data,""),unsafe_allow_html=True)
        else:    
            df_leaderboard = pd.concat([df_leaderboard,pd.DataFrame(results_dict[classifier]['metrics'],index=[classifier])])      
    
    if is_gold_standard :
        st.table(df_leaderboard)        
            

def show_ui_dashboard(app_results,results_dict,option_classifiers,is_gold_standard):
       
    df_a = app_results['data'] 

    features = app_results['comparison_vector']
    index_name = app_results['index_name']
    if is_gold_standard :
        index_name_1 = app_results['index_name_1']
        index_name_2 = app_results['index_name_2']
        df_true_links = app_results['df_true_links']
    
    # For each classifier display results                                                 
    for classifier, results in results_dict.items():
        #st.markdown("### "+classifier)
         
        if  not is_gold_standard :
            pass
        else:      

            if st.checkbox(classifier,key="leader_details"+classifier):
                
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
                    data = get_confusion_matrix_data(selected_data,df_a,
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
                    
    