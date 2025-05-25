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

start_time = time.time()
test_name = "Complete Blood Count (CBC)"
report_path = "samples/sample_report1.jpg"

img_to_table = ImageToTable()
img_to_table.load_image(report_path)
img_to_table.detect_table()
data = img_to_table.extract_table()

print("Data: ", data)

processing_report = ProcessingReport(data, test_name)

all_report_data = processing_report.process()
inference_module = InferenceModule(all_report_data)
inference = inference_module.run()

print("Inference: ", inference)
time_taken = time.time() - start_time
print("Time taken: ", time_taken)