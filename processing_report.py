import transformers
import requests
from requests.auth import HTTPBasicAuth

import torch
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import os, dotenv
from inference import convert_to_json

DB_FAISS_PATH = "vectorstore/reference_range_db"
MEDICAL_MODEL_PATH = "medical_llm/"

dotenv.load_dotenv()
USERNAME=os.getenv('USERNAME')
PASSWORD=os.getenv('PASSWORD')

class ProcessingReport:
    def __init__(self, data, test_name):
        self.data = data
        self.records = self.data['data']
        self.load_models()
        self.test_name = test_name
        self.test = self.get_test_details(test_name)
        # print(self.test)
        self.test_details = self.test['Explanation']
        self.test_type = self.test['Type']
        print(self.test_details)
        print(self.test_type)
        
    def load_models(self):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=MEDICAL_MODEL_PATH,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.embedding_model = HuggingFaceEmbeddings(model_name = "NeuML/pubmedbert-base-embeddings")
        self.db = FAISS.load_local(DB_FAISS_PATH, self.embedding_model, allow_dangerous_deserialization=True)


    def get_inference_over_individual_test(self, test_name, record, context, test_details):
        prompt = f'''You have been given a record from a lab report of {test_name}. You must understand what components are used for this test and then perform the directed task.
        Details of test: {test_details}
        Records: {record}

        Your task is to generate an accurate inference by comparing the result with the expected range provided in the context.
        If you think the following context is not related to the test, but you understand it then answer using your knowledge and ignore context, otherwise you must return Context not enough.
        Context: {context}

        Instructions:
        1. If the record contains only headers or irrelevant information, return None.
        2. If the record describes a test with a non-numerical result, provide a concise and meaningful inference using your medical knowledge.
        3. If the context does not provide enough information to generate a reliable inference, return None.
        4. Do **NOT** make assumptions or add any information that is not present in the context.
        5. Do **NOT** include any disclaimers or additional information in the response.
        6. DO **NOT** hallucinate.

        It should be short under 140 characters and make it consise.
        You must stricktly follow the following format and do not add any additional information and do not explain what and why:
        Format:
        Inference: <inference>
        '''


        messages = [
            {"role": "system", "content": "You are an expert trained on healthcare and Lab laboratory tests domain!"},
            {"role": "user", "content": prompt},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        return outputs[0]["generated_text"][len(prompt):]


    def process(self):
        components = []
        for i in self.records:
            components.append(list(i.keys())[0])

        common_names = []
        for component in components:
            long_common_name, related_names = self.get_common_names(component)
            common_names.append({"long_common_name": long_common_name, "related_names": related_names})


        context = []
        for component in common_names:
            curr = ""
            if not component["related_names"]:
                docs = self.db.similarity_search(component["long_common_name"], k = 2)
            else:
                docs = self.db.similarity_search(component["long_common_name"] + " " + component["related_names"], k = 3)

            for doc in docs:
                curr += doc.page_content + "\n"
            context.append(curr)
        for i in range(len(self.records)):
            self.records[i].update(common_names[i])

        inferences = []
        for i in range(len(self.records)):
            inferences.append(self.get_inference_over_individual_test(self.test_name, self.records[i], context[i], self.test_details))
        for i in range(len(self.records)):
            self.records[i].pop("related_names")
            self.records[i].update({"inference":inferences[i]})

        # yet to add patient details
        return {
                "test_name": self.test_name,
                "test_details": self.test_details,
                "records": self.records,
            }


    @staticmethod
    def get_common_names(component):
        url = f"https://loinc.regenstrief.org/searchapi/loincs?query={component}"
        response = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD))

        if response.status_code == 200:
            res = response.json()
            if res['Results']:
                return res['Results'][0]['LONG_COMMON_NAME'], res['Results'][0]['RELATEDNAMES2']
            else:
                return component, None
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return component, None


    def get_test_details(self, test_name):
        prompt = f"What is {test_name} test? Explain in 2 line."
        messages = [
            {"role": "system", "content": "You are an expert trained on healthcare and Lab laboratory tests domain!"},
            {"role": "user", "content": prompt},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        print(outputs[0]["generated_text"][len(prompt):])
        return convert_to_json(outputs[0]["generated_text"][len(prompt):])
    

# if __name__ == "__main__":
#     data = {'index': ['TEST', 'DIFFERENTIAL LEUCOCYTE COUNT', '(quad) NEUTROPHILS', 'LYMPHOCYTE', 'EOSINOPHILS', 'MONOCYTES', 'BASOPHILS'], 'columns': ['TEST', 'VALUE', 'UNIT', 'REFERENCE'], 'data': [{'DIFFERENTIAL LEUCOCYTE COUNT': {'VALUE': '', 'UNIT': '', 'REFERENCE': ''}}, {'(quad) NEUTROPHILS': {'VALUE': '64', 'UNIT': '(%)', 'REFERENCE': '(40-80)'}}, {'LYMPHOCYTE': {'VALUE': '25', 'UNIT': '(%)', 'REFERENCE': '(20-40)'}}, {'EOSINOPHILS': {'VALUE': '4', 'UNIT': '(%)', 'REFERENCE': '(1-6)'}}, {'MONOCYTES': {'VALUE': '6', 'UNIT': '(%)', 'REFERENCE': '(2-10)'}}, {'BASOPHILS': {'VALUE': '1', 'UNIT': '(%)', 'REFERENCE': '(<2)'}}]}

#     pr = ProcessingReport(data, "Complete Blood Count (CBC)")