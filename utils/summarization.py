from utils.ollama import model
from typing import Optional


def summarize_news(news_text: str,company: str) -> str:
    """Generate concise financial news bullet points with reliable extraction"""
    if not news_text.strip():
        return "• No recent news available"
    
    # Focused prompt for financial news
    prompt = f"""Extract exactly 5 key points from these news articles:
    
    Requirements:
    - Each point must start with •
    - Summarize finacial highlights 
    - 15 words maximum per point
    - Ensure each point is a complete thought
    
    Company: {company}

    News Content:
    {news_text[:3000]}
    
    Extracted Financial Points:
    •"""  # Seed with first bullet

    try:
        response = model.invoke(prompt)
        bullets = []
        current_bullet = ""

        for line in response.split('\n'):
            stripped = line.strip()
            
            if stripped.startswith('•'):
                if current_bullet:
                    bullets.append(current_bullet)
                current_bullet = stripped
            elif current_bullet and stripped:
                current_bullet += " " + stripped
            
            if len(bullets) >= 5:
                break

        if current_bullet and len(bullets) < 5:
            bullets.append(current_bullet)

        if not bullets:
            return "• No financial highlights available"

        return "\n".join(bullets[:5])
    
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        # Provide fallback with raw text highlights
        sentences = [s.strip() for s in news_text.split('. ')[:5] if s.strip()]
        return "\n".join([f"• {s}" for s in sentences]) or "• News summary unavailable"