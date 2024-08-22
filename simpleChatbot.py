from openai import OpenAI

client = OpenAI()

messages = []
system_msg = input("What type of chatbot would you like to create? ")
messages.append({"role":"system","content":f'You are a/an {system_msg} assistant'})

print("Your new assistance is ready!")

while input != "quit()":
    message = input()
    messages.append({"role":"user","content":message})
    response = client.chat.completions.create(model="gpt-4o-mini-2024-07-18",messages = messages)
    reply = response.choices[0].message.content
    messages.append({"role":"assistant","content":reply})
    print("\n" + reply + "\n")