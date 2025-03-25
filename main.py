from agents.search_agent import SearchAgent

def main():
    prompt = "I want to invest in Tesla and Bitcoin. What can you tell me about them?"
    agent = SearchAgent(user_prompt=prompt)
    full_report = agent.process_all()
    
    print("Reporte Consolidado:")
    print(full_report)

if __name__ == "__main__":
    main()