from search_agent import SearchAgent

def main():
    prompt = "Quiero invertir en una empresa de tecnología española."
    agent = SearchAgent(user_prompt=prompt)
    full_report = agent.process_all()
    
    print("Reporte Consolidado:")
    print(full_report)

if __name__ == "__main__":
    main()