import os 
from openai import OpenAI


client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Initialize message history with a system prompt
message_history = [
    {'role': 'system', 'content': 'You are a helpful assistant.'}
]

def ask_llm_studio(user_message):
    """
    Function to send a message to ChatGPT and receive a response, maintaining history.
    This function appends the user's message to the message history, sends the entire history
    to the ChatGPT model, and appends the assistant's reply to the history.
    It returns the assistant's reply.
    This allows for a conversational context to be maintained across multiple interactions.
    This function is useful for creating a conversation between a user and the ChatGPT model.
    
    Args:
        user_message (str): The message to send to ChatGPT.
        
    Returns:
        str: The response from ChatGPT.
    """
    # Add user message to history
    message_history.append({'role': 'user', 'content': user_message})

    try:
        # Send the entire history
        response = client.chat.completions.create(
            model = "gpt-4-1106-preview",
            messages = message_history,
            temperature = 0.7,
            max_tokens = 500,
            n = 1
        )
        # Get the reply and add it to history
        assistant_reply = response.choices[0].message.content
        message_history.append({'role': 'assistant', 'content': assistant_reply})
        return assistant_reply

    except Exception as e:
        print("Error:", e)
        return "An error occurred while processing your request. Please try again later."

if __name__ == "__main__":
    print("Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = ask_llm_studio(user_input)
        print("ChatGPT:", response)
