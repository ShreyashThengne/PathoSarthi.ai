from operator import itemgetter
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import GoogleGenerativeAI
import dotenv, os
import json


dotenv.load_dotenv()
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []



class LLMWithMemory:
    def __init__(self, model_name):
        self.store = {}
        self.model_name = model_name
        self.llm = GoogleGenerativeAI(model=self.model_name, google_api_key=GOOGLE_API_KEY)
        self.set_chain()

    def get_by_session_id(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryHistory()
        return self.store[session_id]

    def set_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You're a pathologist who's good at Interpreting pathology lab reports for domains: Hematology, Biochemistry, Clinical Pathology."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm

        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            self.get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )

    def prompt_llm(self, question):
        return self.chain_with_history.invoke(
            {"question": question},
            config={"configurable": {"session_id": "foo"}}
        )


def convert_to_json(response):
    x = response.split("json\n")[1].split("\n")
    x.pop(0)
    x.pop(-1)
    x = [y.strip() for y in x]

    return json.loads("{" + "".join(x))


HAEMATOLOGY_DB_PATH = "vectorstore/haematology_db"
class InferenceModule:
    def __init__(self, all_report_data):
        self.all_report_data = all_report_data
        self.chances_left = 3
        self.context = ""
        self.id = 0
        self.embedding_model = HuggingFaceEmbeddings(model_name = "NeuML/pubmedbert-base-embeddings")

        self.db = FAISS.load_local(HAEMATOLOGY_DB_PATH, self.embedding_model, allow_dangerous_deserialization=True)

    def add_context(self, query, k = 5):
        docs = self.db.similarity_search(query, k)
        for doc in docs:
            self.context += f"Document {self.id+1}: {doc.page_content}\n"
            self.id += 1

    def generate_context(self):
        llm = LLMWithMemory("gemini-2.0-flash")

        prompt = f"""Read through this lab report, which includes test results and brief inferences. Some inferences present in this report may be wrong so ignore those.
        Lab Report: {self.all_report_data}

        Now, generate a hypothetical document that:
        - Provides possible explanations for abnormal results.
        - References clinical guidelines to interpret the findings.
        - Suggests potential follow-up actions based on the report.

        This document will be used to fetch document from the RAG using HyDE technique.
        Directly start generating the query, no greetings or extra formalities. Return the inference as plain text only. Do not include any explanations, comments, formatting, headings or introduction.
        """

        self.add_context(llm.prompt_llm(prompt))

        while True:
            prompt = f"""
            Here's the context fetched until now:
            Context: {self.context}
            Given the context fetched until now, are you still able to interpret the lab report with good confidence?
            If not, then what extra information do you need? Generate a query which is suitable and have enough text to be used to fetch the information you further need to intepret the result.
            Be very specific and do not ask for patient related information. Only the information which can be fetched from the books. Understand that this query will be feeded to vector database, so generate the query considering how documents are fetched from vector databases.
            Generate a Hypothetical document query for this.
            Answer in this format (only for this prompt), no extra information, no greetings or extra formalities:""" + """
            {
                able_to_interpret: <yes/no>
                extra_info_needed: <extra information needed>
            }
            """

            response_str = llm.prompt_llm(prompt)
            response = convert_to_json(response_str)

            if response["able_to_interpret"] == "yes" or self.chances_left == 0: break

            self.add_context(response['extra_info_needed'], k = 5)

            self.chances_left -= 1

        return self.context

    def run(self):
        self.generate_context()
        llm = LLMWithMemory("gemini-2.5-pro-exp-03-25")

        prompt = f"""
        You have been given a lab report. Interpret it using your knowledge and the context provided. Write your explanation in simple language so that anyone can understand it. Also, mention a conclusion stating what exactly is happening to the person
        Lab Report: {self.all_report_data}

        Context: {self.context}

        Instructions:

        Explain the lab report in plain, everyday terms.
        Identify any results that seem unusual.
        Give a clear summary of what the report means.
        Do not mention any missing information.
        Make sure to pay attention to the units for each value.
        Do not mention from what document you got the information.

        Also, mention remedies. Be specific regarding what could be the suggested diet.

        Output the result in the following format (strictly follow this format):""" + """ { 'interpretation': '<your detailed interpretation>', 'confidence_level': '<high/medium/low>'} """
        response_str = llm.prompt_llm(prompt)
        response = convert_to_json(response_str)
        return response
















"""
Old prompts:
        You have been given a lab report. You need to interpret it using your knowledge and the context provided.

        Lab Report: {self.all_report_data}

        Context: {self.context}

        Instructions:
        - Analyze the lab report thoroughly by correlating the findings with the context.
        - Identify any abnormalities, potential diagnoses, or areas requiring further investigation.
        - Highlight any missing or unclear information that could improve the interpretation.
        - Summarize the interpretation with concise insights while avoiding speculation.
        - Output the result in the following format (strictly follow this format):"""+"""
        {
            'interpretation': '<your detailed interpretation>',
            'confidence_level': '<high/medium/low>'
        }
"""