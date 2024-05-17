 fastApi_llm
This project is to learn FastAPI with LLM

## Contribution
1. Clone from the dev branch 
2. Create a feature branch from the dev branch.
3. Push changes to the feature branch.
4. Create a PR from the feature branch to the dev branch.
5. (Optional) Merge to the main branch only on new release.

## Installation
```bash
pip install virtualenv
python -m venv env
source env/bin/activate

pip install -r requirements.txt

# Model
pip install glfs
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0


# main.py

from transformers import pipeline

model_name = "path/to/TinyLlama-1.1B-Chat-v1.0"
model = pipeline('text-generation', model=model_name)

# Your FastAPI code follows...


#start the streamlit apps
 streamlit run app_1.py

 #start the fastapi apps
 fastapi dev main.py

