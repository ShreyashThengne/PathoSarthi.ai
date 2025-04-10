from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
# import pytesseract
import re
from peft import PeftModel

class ImageToTable:
    def __init__(self):
        self.table_detection_image_processor = AutoImageProcessor.from_pretrained("Table_detection/")
        self.table_detection_model = TableTransformerForObjectDetection.from_pretrained("Table_detection/")

        self.data_extraction_tokenizer = AutoTokenizer.from_pretrained('Table_extraction/', trust_remote_code=True)
        self.data_extraction_model = AutoModel.from_pretrained('Table_extraction/', trust_remote_code=True, use_safetensors=True, pad_token_id=self.data_extraction_tokenizer.eos_token_id)
        self.data_extraction_model = PeftModel.from_pretrained(self.data_extraction_model, "checkpoint-15/")
        # self.data_extraction_model = self.data_extraction_model.eval().cuda()

        adapter_path = "checkpoint-15/"  # this is your folder with adapter_model.safetensors
        model = PeftModel.from_pretrained(self.data_extraction_model, adapter_path)
        self.image = None

        # pytesseract.pytesseract.tesseract_cmd = r'C:\Users\shrey\0_Extra_Folders\tessaract\tesseract.exe'

    def load_image(self, path):
        if not path:
            raise ValueError("No path provided!")

        self.image = Image.open(path).convert("RGB")

    def detect_table(self):
        if not self.image:
            raise ValueError("Please load the image!")

        inputs = self.table_detection_image_processor(images=self.image, return_tensors="pt")
        outputs = self.table_detection_model(**inputs)

        target_sizes = torch.tensor([self.image.size[::-1]])
        results = self.table_detection_image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        table_boxes = []

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            table_boxes.append(box)
            print(
                f"Detected {self.table_detection_model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

        if not table_boxes:
            # raise ValueError("No table detected")
            self.image.save("temp/table.jpg")
            return

        box = table_boxes[0]
        box = [int(coord) for coord in box]
        box[0] -= 10
        box[1] -= 10
        box[2] += 10
        box[3] += 10

        table = self.image.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        table.save("table.jpg")

        draw = ImageDraw.Draw(self.image)
        draw.rectangle(box, fill="black")

    def extract_table(self, col_exist = True):
        replace_patterns1 = [r"\n", r"begin{tabular}", r"end{tabular}", r"\|c", r"\|l", r"\|"]
        replace_patterns2 = [r"\\", r"{", r"}", r"mathrm", r"\n", r":", r"begin{tabular}", r"end{tabular}", r"\|c", r"\|l", r"\|"]

        def clean_text(text, replc_patt):
            return re.sub("|".join(replc_patt), "", text).strip()


        # print(res)
        # res = res.replace("hline", "")
        # rows = res.split("\\")
        try:
            res = self.data_extraction_model.chat_crop(self.data_extraction_tokenizer, "table.jpg", ocr_type='format')
            print(res)
            res = rf"{res}"
            res = res.replace("hline", "")
            rows = res.split(r"\\")[1:]

            for i in range(len(rows)):
                rows[i] = clean_text(rows[i], replace_patterns1).split("&")

                for td_idx in range(len(rows[i])):
                    if "multicolumn" in rows[i][td_idx]:
                        s = rows[i][td_idx].replace("{", " ").replace("}", "").replace("mathrm", "").split(" ", 3)
                        s = [i for i in s if i != ""]
                        num_of_cells_occupied = int(s[1]) - 1

                        rows[i][td_idx] = s[-1].strip().replace("\\", "").strip().replace("mathrm", "")

                        for k in range(num_of_cells_occupied):
                            rows[i].insert(td_idx+1, None)
                    else:
                        rows[i][td_idx] = clean_text(rows[i][td_idx], replace_patterns2)

            rows = res.split("\hline")
            for i in range(len(rows)):
                rows[i] = clean_text(rows[i], replace_patterns1).split("&")

                for td_idx in range(len(rows[i])):
                    if "multicolumn" in rows[i][td_idx]:
                        s = rows[i][td_idx].replace("{", " ").replace("}", "").replace("mathrm", "").replace("\\\\\\", "").split(" ", 3)
                        s = [i for i in s if i != ""]
                        num_of_cells_occupied = int(s[1]) - 1

                        rows[i][td_idx] = s[-1].strip().replace("\\", "").strip().replace("mathrm", "")

                        for k in range(num_of_cells_occupied):
                            rows[i].insert(td_idx+1, None)
                    else:
                        rows[i][td_idx] = clean_text(rows[i][td_idx], replace_patterns2)

            num_cols = [len(r) for r in rows]
            num_cols_dict = {}
            for i in num_cols:
                if i in num_cols_dict:
                    num_cols_dict[i] += 1
                else:
                    num_cols_dict[i] = 1
            num_cols = max(num_cols_dict)

            rows = [r for r in rows if len(r) == num_cols]

            index = [r[0] for r in rows]
            if col_exist:
                columns = rows[0]

                json_table = {"index": index, "columns": columns, "data":[]}
                for i, idx in enumerate(index[1:]):
                    d = {idx: {}}
                    for j, col in enumerate(columns[1:]):
                        d[idx].update({col:rows[i+1][j+1]})
                    json_table['data'].append(d)

            else:
                json_table = {"index": index, "columns": None, "data":[]}
                for i, idx in enumerate(index):
                    d = {idx: []}
                    for j, col in enumerate(rows[i][1:]):
                        d[idx].append(col)
                    json_table['data'].append(d)

            self.free_up_gpu_mem()

            return json_table
        
        except:
            res = self.data_extraction_model.chat_crop(self.data_extraction_tokenizer, "table.jpg", ocr_type='format')
            self.free_up_gpu_mem()
            res = res.replace("\multicolumn", "").replace("{|l|}", "")
            rows = res.split("\hline")
            for i in range(len(rows)):
                rows[i] = clean_text(rows[i], replace_patterns1).split("&")

                for td_idx in range(len(rows[i])):
                    if "multicolumn" in rows[i][td_idx]:
                        s = rows[i][td_idx].replace("{", " ").replace("}", "").replace("mathrm", "").split(" ", 3)
                        s = [i for i in s if i != ""]
                        num_of_cells_occupied = int(s[1]) - 1

                        rows[i][td_idx] = s[-1].strip().replace("\\", "").strip().replace("mathrm", "")

                        for k in range(num_of_cells_occupied):
                            rows[i].insert(td_idx+1, None)
                    else:
                        rows[i][td_idx] = clean_text(rows[i][td_idx], replace_patterns2)

            num_cols = [len(r) for r in rows]
            num_cols_dict = {}
            for i in num_cols:
                if i in num_cols_dict:
                    num_cols_dict[i] += 1
                else:
                    num_cols_dict[i] = 1
            num_cols = max(num_cols_dict)

            rows = [r for r in rows if len(r) == num_cols]

            index = [r[0] for r in rows]
            if col_exist:
                columns = rows[0]

                json_table = {"index": index, "columns": columns, "data":[]}
                for i, idx in enumerate(index[1:]):
                    d = {idx: {}}
                    for j, col in enumerate(columns[1:]):
                        d[idx].update({col:rows[i+1][j+1]})
                    json_table['data'].append(d)
            
            else:
                json_table = {"index": index, "columns": None, "data":[]}
                for i, idx in enumerate(index):
                    d = {idx: []}
                    for j, col in enumerate(rows[i][1:]):
                        d[idx].append(col)
                    json_table['data'].append(d)
                
            return json_table
            

    def free_up_gpu_mem(self):
        import gc
        del self.table_detection_image_processor, self.table_detection_model, self.data_extraction_model, self.data_extraction_tokenizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    report_path = "images/1.png"
    img_to_table = ImageToTable()
    img_to_table.load_image(report_path)
    img_to_table.detect_table()
    data = img_to_table.extract_table()
    print(data)