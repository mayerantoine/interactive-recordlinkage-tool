import sys
import streamlit as st
import recordlinkage as rl
import pandas as pd 
import numpy as np 
import os
from os import path
import altair as alt
from recordlinkage.preprocessing import clean,phonetic
import cProfile
import base64
import logging
from metaphone import doublemetaphone
import matching
from PIL import Image

_compare_vartype = ['exact','string','numeric']
_compare_string_method = ['jarowinkler','jaro','levenshtein', 'damerau_levenshtein', 'qgram','cosine']
_phonetic_encoding = ['metaphone','double_metaphone','soundex','nysiis','match_rating']
_blocking = ['Full Indexing','Standard','SortedNeighbourhood']
_classifiers = ['SimSum','Weighted Average','ECM','Logistic Regression','Naive Bayes','Support Vector Machine']

       
from functools import wraps
import time

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        logging.info("@timefn: %s took  %s  seconds",fn.__name__,str(t2 - t1))
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
 
    
@st.cache(allow_output_mutation=True)
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
    st.info(""" 
             The main task of data cleaning and standardization is the conversion of the raw input data into well defined, consistent forms, as well as the resolution of inconsistencies 
             in the way information is represented and encoded.(Christen 2012)
             
             """)
    for i,c in enumerate(encoding_cols):
        if st.checkbox(c,key="ppp_check"+c) :
            ph_selected = st.selectbox("phonetic encoding:",_phonetic_encoding,key='phonetic_'+c)
            select_encoding[c] = ph_selected
    
    return select_encoding
 
       
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
def show_ui_blocking(cols):
    logging.info("UI- show_ui_blocking...")
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
    logging.info("UI- show_ui_comparison...")
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

  
@timefn    
def show_ui_import_data():
    st.header("Step 1- Let's **begin** with the dataset you want to deduplicate :")
    uploaded_file  = st.file_uploader("Browse and select the file:",type = "csv")
    
    return uploaded_file
    


def show_ui_import_gold_standard(): 
    st.markdown("---")
    st.header("Step 2- Do you have a **gold standard** for this dataset ?")
    st.info("""
            In order to asses the quality of the different matching strategies ground-thruth data , 
                    known as gold standard or labled data are required
            """)
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

@timefn
def show_ui_classification(features,df_a):
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
    for i , classifier in enumerate(_classifiers):
        if st.checkbox(classifier) :
            option_classifiers.append(classifier)
            if classifier == 'SimSum' :
                threshold = st.slider("Threshold :",1,len(features.columns)+1)
            elif classifier == "Weighted Average":
                threshold_wavg = st.number_input("Threshold",min_value=0.1,max_value=0.999,key="threshold_wf")
                cols = features.columns.to_list()
                logging.info("features cols %s:",cols)
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
        
        results_dict[select_classifier]['unique'] = matching.get_unique(df_a,matches,index_name,m[0],m[1])
        results_dict[select_classifier]['metrics']  = {}
        results_dict[select_classifier]['metrics']['nunique'] = results_dict[select_classifier]['unique']['nunique']    
        
        ##FIXME Can we separate metrics calculation from UI render
        if is_gold_standard :
            results_dict[select_classifier]['unique'] = matching.get_unique(df_a,matches,index_name,index_name_1,index_name_2)
            results_dict[select_classifier]['metrics'] =  matching.metrics(df_true_links,matches,features)    
            results_dict[select_classifier]['matrix'] = rl.confusion_matrix(df_true_links,matches,len(features))
            results_dict[select_classifier]['roc'] = matching.show_roc_curve(df_true_links,decision_proba)
            results_dict[select_classifier]['pr'] = matching.show_precision_recall_curve(df_true_links,decision_proba)
            
    return results_dict
                                                              
@timefn                     
def show_ui_leaderboard(app_results,results_dict,option_classifiers,is_gold_standard):
    logging.info("UI- show_ui_leaderboard...")
     
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
            unique_data = results['unique']['data_unique']
            nunique = results['metrics']['nunique']
            st.write("Number of unique records : "+ str(nunique))
            st.write(unique_data)
            st.markdown(get_table_download_link(unique_data,""),unsafe_allow_html=True)
        else:    
            df_leaderboard = pd.concat([df_leaderboard,pd.DataFrame(results_dict[classifier]['metrics'],index=[classifier])])      
    
    st.table(df_leaderboard)        
            

def show_ui_dashboard(app_results,results_dict,option_classifiers,is_gold_standard):
    logging.info("UI- show_ui_dashboard...")
       
    df_a = app_results['data'] 

    features = app_results['comparison_vector']
    index_name = app_results['index_name']
    if is_gold_standard :
        index_name_1 = app_results['index_name_1']
        index_name_2 = app_results['index_name_2']
        df_true_links = app_results['df_true_links']
    
    # For each classifier display results                                                 
    for classifier, results in results_dict.items():
        st.markdown("## "+classifier)
         
        if  not is_gold_standard :
            unique_data = results['unique']['data_unique']
            nunique = results['metrics']['nunique']
            st.write("Number of unique records : "+ str(nunique))
            st.write(unique_data)
            st.markdown(get_table_download_link(unique_data,""),unsafe_allow_html=True)
        else:      

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
                    
                    
def run_app():   
    
    logging.info("running app....")
    st.title("Interactive Record Linkage Tool")
        
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
            st.write("Total number of records :", len(df_a))
            
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
                options_cols,blocking_selected = show_ui_blocking(cols)
                
                if st.checkbox("Run Indexing"):
                    candidate_pairs = matching.run_blocking(df_a,options_cols,blocking_selected)
                    st.markdown('**'+str(len(candidate_pairs))+'** pairs')
                
                    # Comparison
                    comparison = show_ui_comparison(cols,_compare_vartype,_compare_string_method)
        
                    if st.checkbox("Run Comparison"):
                        features = matching.run_comparison(df_a,candidate_pairs,comparison)
                        
                        # UI Classification 
                        option_classifiers, threshold,threshold_wavg,select_wf = show_ui_classification(features,df_a)               
            
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
                                elif select_classifier == 'Logistic Regression':
                                    matches, decision_proba = matching.logreg_classifier(features,df_true_links,train_size=0.2,cv=10)
                                elif select_classifier == "Weighted Average" :
                                    matches,decision_proba = matching.weighted_average_classifier(threshold_wavg,features,select_wf)
                                elif select_classifier == "Naive Bayes":
                                    matches,decision_proba = matching.nb_classifier(features,df_true_links,train_size=0.2,cv=10)
                                elif select_classifier == "Support Vector Machine":
                                    matches,decision_proba = matching.svm_classifier(features,df_true_links,train_size=0.2,cv=10)
                                else:
                                    pass
                                
                        
                                app_results[select_classifier] = {}   
                                app_results[select_classifier]['matches'] = matches
                                app_results[select_classifier]['decision_proba'] = decision_proba        
                            
                            st.markdown("---")
                            st.header('Step 7 -  Evaluation')
                            st.info(""" 
                                    Comparing match results with the known ground truth or gold standard 
                                    to measure the performance of the matching process.
                                    """)
                            if st.checkbox("Run metrics"):
                                results_dict = run_metrics(app_results,option_classifiers,is_gold_standard)
                            
                                if st.checkbox("Show Leaderboard"):
                                    st.markdown('## Leaderboard')
                                    show_ui_leaderboard(app_results,results_dict,option_classifiers,is_gold_standard)
                                if st.checkbox("Show Metrics Dashboard"):
                                    show_ui_dashboard(app_results,results_dict,option_classifiers,is_gold_standard)
        
        logging.info("end running app....")                              


def main():
    """Main function of the App"""    
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(asctime)s %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    
    logging.info("Start...")
    st.sidebar.title("Menu")       
    app_mode = st.sidebar.selectbox("Please select a page", ["Home",
                                                            "Instructions and Guide",
                                                             "Run Deduplication"])
    app_results = {}
    if(app_mode == 'Home'):
        st.title("Interactive Record Linkage Tool")
        st.write("""
                 [Interactive Record Linkage Toolkit](https://github.com/mayerantoine/interactive-recordlinkage-tool) is a tool to experiment 
                 and automatically compare matching quality of different data matching algorithm.
                 This is a prototype  developed using Python and [Streamlit](https://streamlit.io/).
                 
                 This tool is built on top of [Python Record Linkage Toolkit](https://github.com/J535D165/recordlinkage). 
                 
                 This application aspiring goals are to provide :
                  - An interactive way to deduplicate your dataset data whitout coding
                  - A guided and rigorous pair-wise record linkage  steps-by-step process as implemented in the [Python Record Linkage Toolkit](https://github.com/J535D165/recordlinkage)
                  - An intuitive interface to compare the effectiveness of different matching algorithm
                  
                For now , the use case and focus of the tool is on **patient demographic data matching**.

                 """)
    elif(app_mode == "Instructions and Guide"):
        st.title("Instructions and Guide")
        st.header("What is patient matching?")
        st.write(""" 
                 * Data matching : The process to identify records that refer to the same real-world entity within one or across several databases.
                    * When applied on one database, this process is called **deduplication**.
                    * Also known as entity resolution, object identification, duplicate detection.
                 
                 **Patient matching** : Comparing data from multiple sources to identify records that represent the same patient.
                 """)
        st.header("Patient matching process (deduplication)")
        st.write("""
                 * Record linkage is a complex process and requires the user to understand, up to a certain degree, many technical details.
                 * The process consists of 5 steps : **Pre-processing, Indexing/Blocking, Comparison, Classification, Evaluation**
                 * It is important to understand the process because each step influence the matching results
                 * For example understanding blocking strategy is crucial because affecting performance and matching results
                                
                  """)
       
        image = Image.open('matching_process_dedup.PNG')
        st.image(image, caption="record linkage", use_column_width = True)
    elif(app_mode == "Run Deduplication"):
        run_app()
 
             
if __name__ == "__main__":
    main()



