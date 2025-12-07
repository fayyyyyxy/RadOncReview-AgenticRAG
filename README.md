# RadOncReview-AgenticRAG
## How to run the project
### 1. Create and activate a virtual environment
Follow this [guide](https://www.w3schools.com/python/python_virtualenv.asp) to set up a Python virtual environment.
### 2. Install dependencies
After activating the virtual environment, install all needed dependencies by running the following command in the terminal:
```bash
pip install -r requirements.txt
```
### 3. Set up external services
Before running the data preparation pipeline, complete the setup for:
- [Google Docs API](https://github.com/akiqiu8/RadOncReview-AgenticRAG/blob/main/data_prep/setup-guides/GoogleDocsAPI-setup.pdf)
- [AWS Command Line Interface (AWS CLI)](https://github.com/akiqiu8/RadOncReview-AgenticRAG/blob/main/data_prep/setup-guides/AWS-CLI-setup.pdf)
### 4. Run the data preparation pipeline
Once the API and AWS configurations are complete, you can proceed to run the script:
```bash
python data_prep/main.py
```
This script extracts and cleans text from the RadOncReview Google Docs, performs chunking and embedding, and uploads the generated JSON files and FAISS index to AWS (e.g. S3, DynamoDB).
