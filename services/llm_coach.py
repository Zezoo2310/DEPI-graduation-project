import os
import google.generativeai as genai
import json

def get_ai_coach_response(user_message: str, report_data: dict, chat_history: list = None, language: str = "English") -> str:
    """
    Sends the user's message to the Gemini API, along with their interview report context.
    Returns the AI's response text.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "⚠️ **API Key Missing**: The Gemini API key is not configured. Please add `GEMINI_API_KEY` to your environment variables or `.env` file to activate the AI Coach."

    genai.configure(api_key=api_key)
    
    # Initialize the model
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Prepare the context prompt using the user's report
    system_instruction = f"""
    You are the 'HireFlow AI Coach', an expert, empathetic, and encouraging interview coach.
    You are chatting with a user who just completed a mock video interview.
    
    Here is their detailed interview report:
    ---
    Score: {report_data.get('score', 0)}/100
    Overall Summary: {report_data.get('overall_summary', '')}
    Strengths: {', '.join(report_data.get('strengths', []))}
    Needs Improvement: {', '.join(report_data.get('needs_improvement', []))}
    Final Assessment: {report_data.get('final_assessment', '')}
    
    Detailed Analysis Sections:
    """
    
    for section in report_data.get('advice_sections', []):
        system_instruction += f"\n- {section.get('category')} ({section.get('status')}): {section.get('score_detail')}"
        for msg in section.get('messages', []):
            system_instruction += f"\n  * {msg}"
            
    system_instruction += """
    ---
    INSTRUCTIONS:
    1. Answer the user's questions based ONLY on their specific report data and standard interview best practices.
    2. Be conversational, supportive, and direct.
    3. Use Markdown formatting (bolding, bullet points) to make your advice readable.
    4. Keep your responses concise (max 3-4 short paragraphs).
    5. If they ask how to improve a specific weakness mentioned in the report, give them actionable, practical tips.
    6. If the interview language is Arabic, reply in Arabic. Otherwise, reply in English.
    """
    
    # Build messages history for Gemini
    messages = [
        {'role': 'user', 'parts': [system_instruction + "\n\nHello AI Coach!"]}
    ]
    
    # Handle the initial proactive analysis trigger
    if user_message == "GENERATE_INITIAL_ANALYSIS":
        user_message = "I have just opened the dashboard. Please give me a proactive analysis of my interview performance based on the report. Highlight 1 or 2 specific weaknesses (especially from voice or video analysis), offer a quick practical tip, and end by asking me what I'd like to practice or focus on. Do not mention that this was an automated prompt."
    
    # Append past chat history if provided
    if chat_history:
        for msg in chat_history:
            # chat_history format from frontend: {text: "...", sender: "user" | "bot"}
            role = 'model' if msg['sender'] == 'bot' else 'user'
            # Gemini strictly expects 'model' for bot and 'user' for user.
            # Avoid sending empty messages
            if msg['text'].strip():
                messages.append({'role': role, 'parts': [msg['text']]})
            
    # Append the new user message
    messages.append({'role': 'user', 'parts': [user_message]})
    
    try:
        response = model.generate_content(messages)
        return response.text
    except Exception as e:
        return f"❌ **AI Connection Error**: Unable to reach Gemini API. Details: {str(e)}"
