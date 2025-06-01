from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

def ask_lm_studio(user_message):
    response = client.chat.completions.create(      
        model="gemma-3-4b",
        messages=[{"role": "system",
 "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message}],
        temperature=0.7,
        n= 1
        )
    return response.choices[0].message.content     

user = "What is the capital of France?"
response = ask_lm_studio(user)                
print(response)
