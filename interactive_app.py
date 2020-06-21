import sys
import streamlit as st
import recordlinkage as rl
import pandas as pd 
import numpy as np 
import os
from os import path
import altair as alt

import cProfile
import logging
from PIL import Image
from functools import wraps
from utils import timefn

import matching
import frontend 

#FIXME get those list from introspection
_compare_vartype = ['exact','string','numeric']
_compare_string_method = ['jarowinkler','jaro','levenshtein', 'damerau_levenshtein', 'qgram','cosine']
_all_classifiers = ['SimSum','Weighted Average','ECM','Logistic Regression','Naive Bayes','Support Vector Machine']
_non_supervised_classifiers =  ['SimSum','Weighted Average','ECM']
       

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
                                                                        
                
def run_app():   
    
    logging.info("running app....")
    st.title("Interactive Record Linkage Tool")
        
    # UI Import data
    uploaded_file = frontend.show_ui_import_data()    
    
    if uploaded_file is not None:       
        if st.checkbox('Import data') :      
            df_a = load_data(uploaded_file)    
            cols = df_a.columns.to_list()
                       
            index_name = frontend.show_ui_set_index(cols)
            df_a = set_data_index(df_a,index_name)
            
                # Show data to deduplicate
            st.markdown("### First 5 rows of imported data")
            st.write(df_a.head())
            st.write("Total number of records :", len(df_a))
            
            # gold stardard
            uploaded_file_true_links,is_gold_standard = frontend.show_ui_import_gold_standard()
            
            
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
                        index_name_1,index_name_2 = frontend.show_ui_set_index_true_links(cols_true,is_gold_standard)      
        
                        # set index
                        df_true_links = set_data_index_true_links( df_true_links,index_name_1,index_name_2,is_gold_standard)
                        
                        # Show data to deduplicate
                        st.markdown("### First 5 rows of gold standard")
                        st.write(df_true_links.index.to_frame(index=False).head())
                        
            
   
            
            # Set phonetic encoding
            selected_encoding = frontend.show_ui_phonetic_encoding(df_a,index_name)
            if st.checkbox("Run Pre-processing") :
                try:
                     df_a, cols = matching.run_phonetic_encoding(df_a,selected_encoding)
                except AttributeError as error:
                    st.error("Cannot convert attribute type. Please select another field.")
               
                
          
                # Blocking  
                options_cols,blocking_selected = frontend.show_ui_blocking(cols)
                
                if st.checkbox("Run Indexing"):
                    # should be running in background jobs for performance and scalability
                    candidate_pairs = matching.run_blocking(df_a,options_cols,blocking_selected)
                    st.markdown('**'+str(len(candidate_pairs))+'** pairs')
                
                    # Comparison

                    comparison = frontend.show_ui_comparison(cols,_compare_vartype,_compare_string_method)
        
                    if st.checkbox("Run Comparison"):
                        # should be running in background jobs for performance and scalability
                        features = matching.run_comparison(df_a,candidate_pairs,comparison)
                        
                        # UI Classification 
                        if is_gold_standard:
                            option_classifiers, threshold,threshold_wavg,select_wf = frontend.show_ui_classification(features,df_a,_all_classifiers)  
                        else:
                            option_classifiers, threshold,threshold_wavg,select_wf = frontend.show_ui_classification(features,df_a,_non_supervised_classifiers)              
            
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
                            # running classification and put results in app_results
                            # should be running in background jobs for performance and scalabily
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
                            
                            frontend.show_ui_evaluation()
                            if st.checkbox("Run metrics"):
                                results_dict = matching.run_metrics(app_results,option_classifiers,is_gold_standard)
                            
                                #if st.checkbox("Show Leaderboard"):
                                frontend.show_ui_leaderboard(app_results,results_dict,option_classifiers,is_gold_standard)
                                
                                if is_gold_standard :
                                    if st.checkbox("Show Metrics Dashboard"):
                                        frontend.show_ui_dashboard(app_results,results_dict,option_classifiers,is_gold_standard)
        
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
                 This is a proof of concept  developed using Python and [Streamlit](https://streamlit.io/).
                 
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



