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

def main():
    logging.basicConfig(level=logging.INFO)
    st.sidebar.title("Menu")       
    app_mode = st.sidebar.selectbox("Please select a page", ["Show Instructions",
                                                             "Run Deduplication"])
    app_results = {}
    if(app_mode == 'Show Instructions'):
        st.title("App Instructions")
    elif(app_mode == "Run Deduplication"):
        st.title("Interactive Recordlinkage Tool")
        #run_app()

                
if __name__ == "__main__":
    main()