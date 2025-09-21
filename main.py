"""
Main script for processing a medical report image and generating inference.
Workflow:
1. Loads a sample medical report image.
2. Extracts tabular data from the image using the ImageToTable module.
3. Processes the extracted data for a specific test (e.g., Complete Blood Count) using the ProcessingReport module.
4. Runs inference on the processed report data using the InferenceModule.
5. Prints the extracted data, inference results, and total processing time.
Modules used:
- table_extraction.ImageToTable: Handles image loading and table extraction.
- processing_report.ProcessingReport: Processes extracted table data for a given test.
- inference.InferenceModule: Performs inference on processed report data.
Variables:
- test_name: Name of the test to process (e.g., "Complete Blood Count (CBC)").
- report_path: Path to the sample report image.
- data: Extracted table data from the image.
- all_report_data: Processed report data ready for inference.
- inference: Result of the inference module.
- time_taken: Total time taken for the process (in seconds).
"""

from table_extraction import ImageToTable
from processing_report import ProcessingReport
from inference import InferenceModule
import time

import streamlit as st
import pandas as pd
import numpy as np


st.title("PathoSarthi")
st.markdown("#### A generativeAI system that simplifies Pathology lab reports into personalized health insights")

st.write("")
st.write("")
st.write("")
st.write("")
start_time = time.time()
test_name = "Complete Blood Count (CBC)"

input_container = st.empty()
with input_container.container():
    report_path = st.text_input("Enter the path to the report: ")
    analyse_botton = st.button("Analyse")
# report_path = "samples/sample_report1.jpg"

if analyse_botton:
    my_progress_bar = st.progress(0)
    img_to_table = ImageToTable()
    img_to_table.load_image(report_path)

    img_to_table.detect_table()
    my_progress_bar.progress(16)

    data = img_to_table.extract_table()
    my_progress_bar.progress(33)

    processing_report = ProcessingReport(data, test_name)
    my_progress_bar.progress(50)

    all_report_data = processing_report.process()
    my_progress_bar.progress(66)
    
    inference_module = InferenceModule(all_report_data)
    my_progress_bar.progress(83)

    inference = inference_module.run()
    my_progress_bar.progress(100)
    my_progress_bar.empty()
    # input_container.empty()

    st.write(f"**Confidence Level: {inference['confidence_level'].capitalize()}**")
    st.markdown("**Inference:**")
    st.write(inference['interpretation'].replace("**", "").replace("*  ", "\n•"))
    # print("Inference: ", inference)
    time_taken = time.time() - start_time
    st.write("Time taken: ", time_taken)