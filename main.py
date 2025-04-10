from table_extraction import ImageToTable
from processing_report import ProcessingReport
from inference import InferenceModule
import time

start_time = time.time()
report_path = "sample_report1.jpg"

img_to_table = ImageToTable()
img_to_table.load_image(report_path)
img_to_table.detect_table()
data = img_to_table.extract_table()

print("Data: ", data)

test_name = "Complete Blood Count (CBC)"
processing_report = ProcessingReport(data, test_name)

all_report_data = processing_report.process()
inference_module = InferenceModule(all_report_data)
inference = inference_module.run()

print("Inference: ", inference)
time_taken = time.time() - start_time
print("Time taken: ", time_taken)