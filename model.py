
# import torch
# from transformers import pipeline

# pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
# messages = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     },
#     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
# ]
# prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
# print(outputs[0]["generated_text"])


# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
# model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto")
import torch



def predict(prompt,model,tokenizer ):
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


if __name__=="__main__":
    prompt = "How can I call You"
    predict(prompt)
    

# prompt = "How can I call You"

# print(out)
