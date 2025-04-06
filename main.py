from agents.search_agent import SearchAgent

def main(prompt, advanced_mode=False):
    
    # prompt = "I want to invest in Tesla and Bitcoin. What can you tell me about them?"
    agent = SearchAgent(user_prompt=prompt)
    full_report = agent.process_all(advanced_mode)
    print(full_report)
    return full_report

# if __name__ == "__main__":
#     main()