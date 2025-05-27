from utils.ollama import model
from typing import Optional


def summarize_news(news_text: str) -> str:
    """Generate 5 bullet points from raw news text using LLM"""
    if not news_text.strip():
        return "No recent news available"
    
    
    prompt = f"""For each news article, create 1-5 concise bullet points:
    
    News Content:
    {news_text}
    
    Format rules:
    • Max 5 bullets in total
    • 15 words maximum per bullet
    • Start each bullet with •
    • Prioritize key facts/impacts
    • Mention sources when available
    • No markdown or extra formatting
    
    Processed Bullet Points:"""

    try:
        response = model.invoke(prompt)
        # Clean up the response
        bullets = [line.strip() for line in response.split('\n') if line.strip().startswith('•')][:5]  
        
        # Fallback if format not followed
        if len(bullets) < 5:
            return "\n".join([f"• {line}" for line in response.split('. ')[:5]])
            
        return "\n".join(bullets)
    
    except Exception as e:
        print(f"News summarization error: {e}")
        return "• Key news points unavailable (summary failed)"
