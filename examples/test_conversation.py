# Does Paraphrasing gives better quality output?
import openai

# Need:
# 1. Dataset of prompts and responses
# 2. LLM
# 3. LLM prompt template
# 4. LLM score aggregation function
# 5. LLM score comparison function


# Begin coomunication with AI coach
# three party: coach, trainee, and judge

print('Setting up a three-party Conversation')
print('-'*80)
print('Insurance Agent is the trainee, tries to sell insurance product to a client')
print('Citizen is the client, tries to learn about insurance product, and decide whether to buy it')
print('AI Coach is the judge, tries to give advice to the trainee on how to improve his sales strategy')

# Google Gemini API key for free
import os
GOOGLE_API_KEY = 'AIzaSyCIW8aSqq3t9aasOdhDAOzXJtKS9KX6j5s'
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

genai.configure(api_key=GOOGLE_API_KEY)

llm = genai.GenerativeModel('gemini-pro')

# response = llm.generate_content("What is the meaning of life?")

# print('Response obtained')
# print(response.text)

global_prompt = "Only respond in your own role, do not respond as other roles. This will be a conversation, so do not go on forever."

# AI coach
coach_system_prompt = "You are a AI Coach named Bob, you train insurance agent to sell insurance products to Singapore client. You inspect the conversation between AI trainee and AI citizen, and give instruction on how AI trainee can improve his saling strategy. Respond in context. \n"
coach_system_prompt += "Respond folloing the format: Coach: [instruction] \n"
# AI sales agent
sales_system_prompt = "You are a insurance agent named Sarah, trying to sale your product to a Singaporen citizen. A trainer will be inspecting your conversation and provide you with adivce, try to improve your sales strategy based on the advice, and try to sell the product to the citizen. \n"
sales_system_prompt += "Respond folloing the format: Sales: [response] \n"
# AI citizen
citizen_system_prompt = "You are a common Singapore citizen, you wish to know more about insurance products in order to decide whether it fits your needs. \n"
citizen_info_prompt_template = "Your gender is {gender}, name is {name}, age is {age}, and your annual income is {income} SGD, you work as {work}, you have {family_size} family members, and you have {health_condition}"
# Example of using the template with a dictionary:
citizen_info = {
    'name': 'Dave',
    'gender': 'male',
    'age': 30,
    'income': '40,000',
    'work': 'computer engineer',
    'family_size': 4,
    'health_condition': 'no chronic diseases'
}
citizen_info_prompt = citizen_info_prompt_template.format(**citizen_info)
citizen_prompt = citizen_system_prompt + citizen_info_prompt
citizen_system_prompt += "Respond folloing the format: Citizen: [question] \n"

# Conversation Initialization
def format_prompt(prompt, system_prompt=""):
    if isinstance(prompt, list):
        prompt = "\n".join(prompt)
    if system_prompt.strip():
        return f"[INST] {system_prompt} {prompt} [/INST]"
    return f"[INST] {prompt} [/INST]"

# Sales initialize converstion
# -- words pass onto everybody, and they react to it
# -- what order to pass? what order to react?
# -- how to record the conversation in memory?

# dumb taking turn response system


print('-'*80)
print('Test with Prompt')
import time
start = time.time()
sales_initial = "Sales: Hello, I am an insurance agent, I am here to help you with your insurance needs."

# modeling with Agent can be useful: 
class Agent:
    def __init__(self, name, system_prompt, llm):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.memory = []
    
    def respond(self):
        start = time.time()
        max_retry = 3
        while max_retry > 0:
            try:
                message = self.llm.generate_content(format_prompt(self.memory, self.system_prompt)).text
                max_retry -= 1
                break
            except:
                max_retry -= 1        
        end = time.time()
        print(f'---- {self.name} respond to message takes {end - start} seconds with {3 - max_retry} calls \n')
        return message

    # agent not really pay attention to everything -- selection should happens in the conversation
    def listen(self, message):
        # print(f'---- {self.name} listen to message')
        self.memory.append(message)

    def forget(self):
        pass # forget about things that are not important | or, the more we use, the more weight we assign, and forget is based on probability (?)
    
    def __str__(self):
        return self.name
    
sales = Agent('Sales', sales_system_prompt + global_prompt, llm)
coach = Agent('Coach', coach_system_prompt + global_prompt, llm)
citizen = Agent('Citizen', citizen_system_prompt + global_prompt, llm)

sales_initial = "Sales: Hello, I am an insurance agent, I am here to help you with your insurance needs. \n"
citizen_decision_prompt = "Make a decision on whether to buy the insurance product, and rate the sales agent on a scale of 1 to 5. \n Format: Citizen: [decision] [rating] \n"
coach_summary_prompt = "Summarize the agent's performance based on concrete context, citizen's feedback and decision, give advice on what part can be improved, and rate the sales agent on a scale of 1 to 5. \n Format: Coach: [summary] [rating] \n"
sales_reflection_prompt = "Reflect on your performance and summarize what you've learned which can improve your performance, what do you think jeopardize your success in selling the product, based on the advice given by the coach, and the decision and rating given by the citizen. Give yourself a rating between 1 to 5.  \n Format: Sales: [reflection] [rating]\n"

terminate = False
respond = ''
n = 0
max_n = 3
while not terminate:
    n += 1

    for i in range(2):
        # Sales respond
        if respond == '':
            respond = sales_initial
        else:
            respond = sales.respond()
            citizen.listen(respond)
            coach.listen(respond)
        
        print(respond)
        # Citizen respond
        if n == max_n:
            # termination reached, ask the citizen to make a decision and rate the sales agent
            citizen.system_prompt += citizen_decision_prompt
            sales.system_prompt += citizen_decision_prompt
            coach_summary_prompt += coach_summary_prompt
            print('-'*80)
            print('Conversation terminating || Citizen making decision || Coach Summarize || Agent Reflect')
        else:
            respond = citizen.respond()
            sales.listen(respond)
            coach.listen(respond)
            print(respond)

        # Coach respond
        respond = coach.respond()
        sales.listen(respond)
        print(respond)

    
    if n == max_n:
        # sales reflection
        respond = sales.respond()
        print(respond)
        terminate = True

    


