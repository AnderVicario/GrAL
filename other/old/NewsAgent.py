import logging
import json
import re
import os
from dotenv import load_dotenv
from datetime import datetime
from together import Together
from gnews import GNews

load_dotenv()
logging.basicConfig(level=logging.INFO)

def fetch_news_articles(query):
    """Fetch news articles for a given query using GNews."""
    try:
        logging.info(f"Fetching articles for query: '{query}'")
        news_client = GNews(
            language='en',
            country='US',
            period='7d',
            max_results=10
        )
        articles = news_client.get_news(query)
        logging.info(f"Fetched {len(articles)} articles")
        return articles
    except Exception as e:
        logging.error(f"Error fetching articles for query: '{query}'. Error: {e}")
        return []


def generate_query_combinations(company_names, keywords):
    """Generate search queries combining company names and keywords."""
    queries = []
    for company in company_names:
        for keyword in keywords:
            queries.append(f"{company} {keyword}")
    return queries


def save_news_data(articles_data, filename="news_results.json"):
    """Save news results to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(articles_data, f, indent=4, ensure_ascii=False)


class FinancialNewsAgent:
    """Agent to generate financial analysis and search for relevant news articles."""
    def __init__(self):
        self.llm_client = Together()
        self.current_date = datetime.today().isoformat()


    def generate_financial_analysis(self, context, question, model_name):
        """Generate financial analysis using LLM."""
        prompt = f"""
You are a highly specialized financial assistant. For every financial query, your response MUST consist ONLY of exactly three parts, in the exact order below, with no additional text or commentary:

1. **Company LIST**: Provide the official company name(s) associated with the question. If the user does not provide any company, it will generate some automatic suggestions of company names based on the context separated with commas.
2. **Expiration Date**: Provide the expiration date for the information's relevance, relative to today's date, in the format YYYY/MM/DD. Use the following rules based on today's date, which is provided:
   - If the query indicates immediate or short-term information (e.g., contains words such as "today", "current", "immediate", "short term", "24 hours"), set the expiration date to tomorrow's date.
   - If the query indicates a long-term period (e.g., contains phrases like "5 years", "long term", "extended period"), set the expiration date to today's date plus the specified period (e.g., if "5 years" is mentioned, add 5 years).
   - If the query requests historical data (e.g., contains "historical", "past", or refers to a past date), set the expiration date to a date in the past (e.g., yesterday's date) to indicate that the information is no longer current.
3. **News & Article Keywords**: Provide concise keyword-style search terms (no full sentences) to retrieve relevant news or reports via news APIs. Each keyword or keyword combination should be on a new line and relevant to the question.

Assume today's date is: {self.current_date}.

Example:
Question: Should I invest in Tesla today?
Answer:
1. TESLA, SPACEX
2. 2025/03/16
3. - stock  
   - shares  
   - investment  
   - financial news  
   - market performance

Now, based on the following context and question, provide your answer following the above structure exactly, with no additional information.

Context:
{context}

Question:
{question}

Answer:
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=2056,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<｜end▁of▁sentence｜>"],
            stream=True
        )
        full_response = ""
        for token in response:
            if hasattr(token, 'choices'):
                content = token.choices[0].delta.content
                full_response += content
        return full_response
    

class SentimentAnalysisAgent:
    def __init__(self):
        self.llm_client = Together()

    def analyze_batch_sentiment(self, titles, question, model_name, batch_size=5):
        """
        Analyze sentiment for a batch of article titles.
        
        Parameters:
        - titles: List of article titles to analyze
        - question: The original financial question for context
        - model_name: Language model to use
        - batch_size: Number of titles to process in each API call
        
        Returns:
        - List of sentiment classifications in the same order as input titles
        """
        all_sentiments = []
        
        # Process titles in batches
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i+batch_size]
            batch_text = "\n".join([f"{j+1}. {title}" for j, title in enumerate(batch)])
            
            prompt = f"""
You are a financial sentiment analysis agent. Your task is to analyze the sentiment of financial news headlines in relation to the following question:
{question}

For each headline below, classify its sentiment as Positive, Neutral (if it is unknown), or Negative based on its potential impact on the financial markets and the investor's interests.

Respond with ONLY a numbered list matching the input format, with just the sentiment classification (Positive, Neutral, or Negative) for each headline.

Headlines to classify:
{batch_text}

Answer:
"""
            logging.info(f"Processing batch of {len(batch)} headlines for sentiment analysis")
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=2056,
                temperature=0.1,  # Lower temperature for more consistent classification
                top_p=0.7,
                top_k=50,
                repetition_penalty=1,
                stop=["<｜end▁of▁sentence｜>"],
                stream=True
            )
            
            full_response = ""
            for token in response:
                if hasattr(token, 'choices'):
                    content = token.choices[0].delta.content
                    full_response += content

            clean_text = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
            
            # Parse the response to extract sentiments
            sentiment_lines = [line.strip() for line in clean_text.strip().split('\n') if line.strip()]
            batch_sentiments = []
            
            for line in sentiment_lines:
                # Extract just the sentiment classification (Positive, Neutral, or Negative)
                sentiment_match = re.search(r'(Positive|Neutral|Negative)', line)
                if sentiment_match:
                    batch_sentiments.append(sentiment_match.group(1))
                else:
                    batch_sentiments.append("Unknown")
            
            all_sentiments.extend(batch_sentiments)
            logging.info(f"Processed batch with sentiments: {batch_sentiments}")
        
        return all_sentiments


def parse_llm_response(response_text):
    """Parse the LLM response to extract companies, date, and keywords."""
    clean_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    pattern = r"1\. (.+?)\s*2\. (.+?)\s*3\. (.+)"
    match = re.search(pattern, clean_text, re.DOTALL)
    
    if match:
        companies = match.group(1).strip()
        expiration_date = match.group(2).strip()
        keywords = match.group(3).strip()
        
        company_list = [company.strip() for company in companies.split(",")]
        keyword_list = [line.strip("- ").strip() for line in keywords.split("\n") if line.strip()]
        
        return clean_text, company_list, expiration_date, keyword_list
    return clean_text, None, None, None


def search_news(companies, keywords, use_advanced_search=False, max_advanced_articles_per_company=10):
    """
    Search for news articles with separate limits for basic and advanced search.
    
    Parameters:
    - companies: List of company names to search for
    - keywords: List of keywords for advanced search
    - use_advanced_search: Boolean flag to enable/disable advanced search
    - max_basic_articles: Maximum total number of articles to collect in basic search (default: 20)
    - max_advanced_articles_per_company: Maximum articles per company in advanced search (default: 10)
    
    Returns:
    - Dictionary with company names as keys and their associated articles as values
    """
    # Use a dictionary to organize articles by company
    company_articles = {company: [] for company in companies}
    
    # Basic search: Search for articles using company names only, with a global limit
    for company in companies:
        logging.info(f"Processing company-only query: {company}")
        articles = fetch_news_articles(company)
        for article in articles:
            company_articles[company].append({
                "query": company,
                "title": article.get("title"),
                "description": article.get("description"),
                "url": article.get("url"),
                "publishedDate": article.get("published date"),
                "sentiment": None  # Will be filled later
            })
    
    # Advanced search: Search for articles using company + keyword combinations, with PER-COMPANY limits
    if use_advanced_search:
        logging.info("Advanced search enabled. Generating query combinations...")
        for company in companies:
            advanced_articles_for_company = 0
            company_keywords = [f"{company} {keyword}" for keyword in keywords]
            
            for query in company_keywords:
                logging.info(f"Processing combined query: {query}")
                articles = fetch_news_articles(query)
                
                for article in articles:
                    company_articles[company].append({
                        "query": query,
                        "title": article.get("title"),
                        "description": article.get("description"),
                        "url": article.get("url"),
                        "publishedDate": article.get("published date"),
                        "sentiment": None  # Will be filled later
                    })
                    advanced_articles_for_company += 1
                    
                    # Check if we've reached the per-company limit for advanced search
                    if advanced_articles_for_company >= max_advanced_articles_per_company:
                        logging.info(f"Reached maximum advanced article limit of {max_advanced_articles_per_company} for company {company}. Moving to next company.")
                        break
                
                # Break from keyword loop if company limit reached
                if advanced_articles_for_company >= max_advanced_articles_per_company:
                    break
    else:
        logging.info("Advanced search disabled. Using basic search only.")
    
    return company_articles


if __name__ == "__main__":
    # Configuration
    context = "Current financial markets, recent economic news, and investment analysis."
    question = "Bitcoin investment this month, should I invest? How is market?"
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    use_advanced_search = False
    max_advanced_articles_per_company = 20
    
    # Initialize and run the news finder agent
    financial_agent = FinancialNewsAgent()
    response = financial_agent.generate_financial_analysis(context, question, model_name)
    
    # Parse the response
    full_output, companies, expiration_date, keywords = parse_llm_response(response)
    
    print(full_output)
    print("\n")
    
    if companies:
        logging.info(f"Companies: {companies}")
        logging.info(f"Expiration Date: {expiration_date}")
        logging.info(f"Search Keywords: {keywords}")
        
        # Search for articles with toggle for advanced search
        company_articles = search_news(companies, keywords, use_advanced_search, max_advanced_articles_per_company)
        
        # Initialize sentiment analysis agent
        sentiment_agent = SentimentAnalysisAgent()
        
        # Process sentiment for each company's articles
        for company, articles in company_articles.items():
            if articles:
                logging.info(f"Analyzing sentiment for {len(articles)} articles about {company}")
                
                # Extract titles for sentiment analysis
                titles = [article["title"] for article in articles if article.get("title")]
                
                # Get sentiment classifications
                sentiments = sentiment_agent.analyze_batch_sentiment(titles, question, model_name)
                
                # Add sentiment to articles
                for i, article in enumerate(articles):
                    if i < len(sentiments):
                        article["sentiment"] = sentiments[i]
        
        # Flatten the dictionary for saving
        all_articles = []
        for company, articles in company_articles.items():
            for article in articles:
                # Add company field to each article for better organization
                article["company"] = company
                all_articles.append(article)
        
        # Save the results
        save_news_data(all_articles)
        logging.info(f"Saved {len(all_articles)} articles with sentiment analysis to news_results.json")
        
        # Print summary of sentiment analysis by company
        print("\nSentiment Analysis Summary:")
        for company, articles in company_articles.items():
            if articles:
                sentiments = [a.get("sentiment") for a in articles if a.get("sentiment")]
                positive_count = sentiments.count("Positive")
                neutral_count = sentiments.count("Neutral")
                negative_count = sentiments.count("Negative")
                
                print(f"\n{company}:")
                print(f"  Positive: {positive_count} ({(positive_count/len(sentiments)*100):.1f}%)")
                print(f"  Neutral: {neutral_count} ({(neutral_count/len(sentiments)*100):.1f}%)")
                print(f"  Negative: {negative_count} ({(negative_count/len(sentiments)*100):.1f}%)")
    else:
        logging.error("Could not parse response format.")