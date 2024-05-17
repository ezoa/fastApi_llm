from fastapi import FastAPI,BackgroundTasks,HTTPException
from fastapi.encoders import jsonable_encoder
import uuid
# from model import predict

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch  
import json

ai_models={}
answers={}



@asynccontextmanager
async def lifespan(app:FastAPI):

     global model_name, tokenizer,model
     # Load model directly
     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
     tokenizer = AutoTokenizer.from_pretrained(model_name)
     model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")
     ai_models["tokenizer"]=tokenizer
     ai_models["model"]=model
     yield
     ai_models.clear()


class InputData(BaseModel):
     
     prompt: str
   


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def predict(prompt):
    input_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False,
    return_tensors="pt"
    )
    tokens = model.generate(
    input_ids.to(device=model.device),
    max_new_tokens=128,
    temperature=0.99,
    top_p=0.95,
    do_sample=True,
    )

    out = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return out





@app.post("/predict")
 
async def handle_prompt(data:InputData,background_tasks:BackgroundTasks):
     """ ML function"""

     prompt=data.prompt.strip()
     task_id=str(uuid.uuid4())
     background_tasks.add_task(

          perform_answer_and_store,
          task_id,
          
          prompt,
     )

     return task_id.strip()


     #json_compatible=predict(data.prompt.strip())
     # data=json.loads(json_compatible)
     # cleaned_string = json_compatible.strip()

     # if json_compatible.startswith("b'") and json_compatible.endswith("'"):
     #      cleaned_string=json_compatible[2,-1]
     #      print(cleaned_string)
     # else:
     #      cleaned_string=json_compatible


     # cleaned_string=cleaned_string.strip('"')


     # result={"langage":"English",
     #         'prompt':data.prompt,
     #         "output": json_compatible
     #         #JSONResponse(content=json_compatible) 

          

     # }
     # # print(type(result))
     # string_data=json.dumps(result)
     # # json_data=json.loads(string_data)
     # cleaned_string = json_compatible.strip('")

     


     # return string_data
     # result=json_compatible
     # return  json_compatible

async def perform_answer_and_store(task_id,prompt):
     answer= await predict(prompt)
     print(f"task id comming to perform_answer_and_store :{task_id}")
     answers[task_id]=answer


@app.get("/result/{task_id}")
async def get_result(task_id :str):
     task_id=task_id.strip()
     print(f"task id after stripping:{task_id}")
     if task_id not in answers:

          raise HTTPException(status_code=404, detail=f"result from model,{answers}")
     return answers[task_id]




   
    
