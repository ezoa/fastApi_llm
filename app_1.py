import streamlit as st
import requests



def call_api(text):

    data= {
            "prompt": text,
            }
    
    try:
        reponse=requests.post("http://127.0.0.1:8000/predict",json=data)
        
        converted=reponse.content.decode("utf-8")
        a=converted.replace('"'," ")
        b=a.replace("\\n"," ")
        st.write(f"task_id:{b}")
        response_2=requests.get(f"http://127.0.0.1:8000/result/{b}")
        
        # return response_2.content

        result=response_2.content.decode("utf-8")
        c=result.replace('"'," ")
        

        return c
        
        # return (type(reponse.content))
        
           
    except  Exception as e:  
         
         print(f"something wrong happen{e}")
         

    
   
    





st.title ("Ezoa Test ")

result=None
input =st.sidebar.text_input("Enter text")

button =st.sidebar.button("start")

if button :
    #call api

    result=call_api(input)
    
print(type(result)) 
st.text_area("output",result)   
print(type(result)) 
    #print response












