<h1 align="center" id="title">PathoSarthi.ai</h1>

<p align="center"><img src="https://socialify.git.ci/ShreyashThengne/PathoSarthi.ai/image?custom_description=A+generative+AI+system+that+simplifies+Pathology+lab+reports+into+personalized+health+insights.&amp;description=1&amp;font=Source+Code+Pro&amp;issues=1&amp;name=1&amp;pattern=Formal+Invitation&amp;pulls=1&amp;stargazers=1&amp;theme=Auto" alt="project-image"></p>

<p id="description">Our product uses generative AI to interpret laboratory reports by extracting standardizing and contextualizing medical data. It converts raw diagnostic content into structured insights and personalized summaries enabling users to easily understand their health information. By simplifying complex medical language and offering relevant guidance the system empowers individuals to make informed decisions and engage more confidently in their healthcare journey.</p>
(This is my work in the project. I didnt work on website, so didnt include it. Project is not deployed on cloud.)
<h2>🛠️ Installation Steps:</h2>

<p>1. Create a New Conda Environment</p>

```
conda create -p .\venv
```

<p>2. Activate the environment</p>

```
conda activate .\venv
```

<p>3. Install requirements</p>

```
pip install -r requirements.txt
```

<p>4. Install PyTorch with Cuda Support</p>

```
torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

<p>5. Download all the required models</p>
Following 3 models needs to be download in a specific path: <br>
1. Microsoft Table Transformer Detection (Model ID: microsoft/table-transformer-detection, Local Directory: table_detection/) <br>
2. GOT-OCR 2.0 (Model ID: stepfun-ai/GOT-OCR2_0, Local Directory: table_extraction/) <br>
3. Bio Medical Llama 3.2 1B with Chain of Thought (Model ID: ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025, Local Directory: medical_llm/) <br>
<br>

```
from huggingface_hub import snapshot_download
snapshot_download(repo_id="model_id" local_dir = dir)
```

<p>6. Create a .env file and put hugging face token (HF_TOKEN) and gemini api token (GOOGLE_API_KEY) in it.</p>

<p>7. Create a LOINC account using this link (https://loinc.org/join/)). Now, put username (USERNAME) and password (PASSWORD) for LOINC in .env file which you just created.</p>
  
  
<h2>💻 Built with</h2>

Technologies used in the project:

*   RAG
*   Gen AI
*   OCR
*   NLP
*   Python

<h2>Working</h2>
We begin by inputting the lab report image into the system. The system sends this image to a table detector, which identifies and crops the image to isolate only the table. This cropped table image is then passed to a table extractor model, which extracts the table in LaTeX format. We then convert this LaTeX into JSON to make it more structured and easier to work with.
<br><br>
Next, we process the lab report data. We extract each tested component from the report and retrieve its common and related names. This is necessary because the same component may be referred to by different names. To standardize and avoid confusion, we use common names with the help of LOINC (Logical Observation Identifiers Names and Codes).
<br><br>
Using these standardized names, we fetch documents that likely include the reference ranges for each component via Retrieval-Augmented Generation (RAG). These documents, along with the component’s common name and the original name from the lab report, are sent to the LLM. We then ask it to generate an inference for each relevant component. These inferences are added to the JSON object created earlier for further processing.
<br><br>
The next stage is the inference module, which is divided into two parts:
<h4>1. Context Collection:</h4>
Here, we collect the knowledge the LLM needs to interpret the report. We start by generating a HyDE (Hypothetical Document Embedding) query, which creates a mock interpretation of the report. This is used to retrieve related documents from medical textbooks using RAG. These documents form the initial context.
<br>
We then provide the processed report and the fetched documents to the LLM and ask if the context is sufficient. If not, the LLM generates new queries to retrieve additional documents. This is a recursive process that continues until the LLM either has enough context or exhausts its attempts.
<br>
<h4>2. Inference:</h4>
With the full context and the processed report, we ask the LLM to analyze the lab report and provide a detailed conclusion. We also ask it to suggest possible remedies and dietary recommendations for the patient.


<h2>🛠️ Project Sample Output:</h2>
<h4> Input: </h4>
<img src="https://github.com/ShreyashThengne/PathoSarthi.ai/blob/8a8f7bfb109c6046a6ee9d04133ed7d3a69d7864/samples/sample_report1.jpg">

<h4> Output:</h4> 
Interpretation:
<br>
"Here's a simple explanation of your blood test results, called a Complete Blood Count (CBC): <br>  <br> What is a CBC? <br> This test gives a general overview of your blood's health. It checks different types of cells in your blood: red blood cells (which carry oxygen), white blood cells (which fight infections), and platelets (which help your blood clot). <br>  <br> Your Results Explained: <br>  <br> 1.  Red Blood Cells (RBCs): <br> * Your Total RBC Count (5 million/cumm), Hematocrit (42%), MCV (Mean Corpuscular Volume, 84.0 fL), and MCH (Mean Cell Hemoglobin, 30.0 Pg) are all within the normal range. This means your body is making enough red blood cells, and they are generally of the correct size and carry enough oxygen, suggesting you do not have common types of anemia. <br> * However, your MCHC (Mean Cell Hemoglobin Concentration), which indicates how concentrated the hemoglobin is inside your red blood cells, is indicated as high. This is an unusual finding. Sometimes, this can be related to how red blood cells are shaped or how much hemoglobin they contain, or it could be influenced by your hydration status. <br>  <br> 2.  White Blood Cells (WBCs): <br> * Your Total Leukocyte Count (5,100 cumm) is normal. This is the overall number of infection-fighting cells. <br> * Looking at the different types of white blood cells: <br> * Your Neutrophils (79%), Eosinophils (1%), and Basophils (1%) are all within their normal ranges. <br> * However, your Lymphocytes and Monocytes are indicated as low. Lymphocytes are important for fighting off viral infections and for your long-term immune memory, while monocytes help clean up damaged cells and fight off certain types of infections. When these specific white blood cells are low, it can sometimes mean your immune system might be a bit weakened in certain areas, or it could be a temporary change due to things like a recent infection or certain medications. <br>  <br> 3.  Platelets: <br> * Your Platelet Count (3.5 lakhs/cumm) is normal. These are the cells that help your blood clot, so a normal count means your blood clotting ability is likely fine. <br>  <br> Conclusion: What exactly is happening to the person? <br> Your blood report shows a mixed picture. Most of your blood counts are normal, which is good. However, two main points stand out: <br>  <br> 1.  Low Lymphocytes and Monocytes: This suggests a mild imbalance in your immune system, where specific types of white blood cells are present in lower numbers than usual. This can be a temporary response to a recent infection (even a mild one like a common cold), certain medications, or stress. Less commonly, it could indicate an underlying issue with your bone marrow. <br> 2.  High MCHC: This is an unusual finding for your red blood cells. While it doesn't always point to a serious problem, especially if other red cell parameters are normal, it's something to note. It can sometimes be seen with certain red blood cell conditions or even severe dehydration. <br>  <br> Overall, you are not showing signs of severe anemia or a major active infection based on these results. The findings suggest a need to monitor your immune cell levels and the MCHC. It's not a cause for immediate alarm but warrants attention to your general health and immune support. <br>  <br> Suggested Remedies: <br>  <br> Since the findings point towards supporting your immune system and general health, here are some recommendations: <br>  <br> 1.  Boost Your Immune System through Diet: <br>     *   Focus on a Rainbow of Fruits and Vegetables: Eat a wide variety of colorful fruits (like berries, citrus fruits, kiwi) and vegetables (like leafy greens, broccoli, bell peppers, carrots, sweet potatoes). These are packed with vitamins (especially Vitamin C, A, E) and antioxidants that support immune health. <br> * Include Lean Proteins: Incorporate sources like chicken, fish, beans, lentils, and tofu. Protein is essential for building and repairing immune cells. <br> * Choose Whole Grains: Opt for oats, brown rice, quinoa, and whole-wheat bread for complex carbohydrates and B vitamins, which provide energy and support cell function. <br> * Healthy Fats are Key: Add avocados, nuts, seeds (like chia and flaxseeds), and olive oil to your diet. Omega-3 fatty acids, particularly found in fatty fish like salmon, are known to support immune regulation. <br> * Zinc-Rich Foods: Zinc is vital for immune function. Include foods like lean meat, poultry, beans, nuts, and dairy. <br> * Ensure Adequate Vitamin D: Get enough Vitamin D through fortified foods, fatty fish, or safe sun exposure, as it plays a crucial role in immune modulation. <br> 2.  Stay Well Hydrated: Drink plenty of water throughout the day. This is good for overall health and can also influence blood parameters like MCHC. <br>  <br> 3.  Prioritize Sleep: Aim for 7-9 hours of quality sleep each night. Your immune system repairs and regenerates during sleep. <br>  <br> 4.  Manage Stress: Chronic stress can weaken your immune system. Practice stress-reducing activities like meditation, yoga, deep breathing exercises, or hobbies you enjoy. <br>  <br> 5.  Avoid Harmful Substances: Limit or avoid smoking and excessive alcohol consumption, as they can negatively impact your immune function."
<br>
<br>
Confidence_level: "high"
