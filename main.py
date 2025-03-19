from search_agent import SearchAgent

def main():
    prompt = "Quiero invertir en una empresa de tecnología española."
    agent = SearchAgent(user_prompt=prompt)
    full_report = agent.process_all()
    
    print("Reporte Consolidado:")
    for key, value in full_report.items():
        print(f"\n{key}: {value}")

if __name__ == "__main__":
    main()