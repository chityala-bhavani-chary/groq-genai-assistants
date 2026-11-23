# Groq GenAI Assistants

Streamlit apps powered by the **Groq** API for chat and CSV data analysis.

## Apps

| File | What it does |
|------|----------------|
| `groq_ai.py` | Simple conversational chatbot |
| `groq_full_ai.py` | Richer chat UI with file helpers |
| `DA_Ai_Assis1.py` | Upload CSV → ask natural-language data questions |
| `DA_Ai_Assis2.py` | Improved Data Analysis AI Assistant |

## Security

API keys are **not** hardcoded. Set:

```bash
export GROQ_API_KEY='your_groq_api_key'
```

Get a key from [Groq Console](https://console.groq.com).

If the key is missing, each app shows setup instructions and exits cleanly.

## Setup

```bash
git clone https://github.com/<your-username>/groq-genai-assistants.git
cd groq-genai-assistants
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY='your_key'
```

## Run

```bash
streamlit run groq_ai.py
streamlit run groq_full_ai.py
streamlit run DA_Ai_Assis1.py
streamlit run DA_Ai_Assis2.py
```

## Data Analysis Assistant flow

1. Upload a CSV  
2. Preview rows / basic summary  
3. Ask questions in plain English  
4. Groq LLM answers using dataset context  

## Tech stack

- Streamlit  
- Groq Python SDK  
- pandas, matplotlib  
- pypdf (for `groq_full_ai.py`)  

Default model in scripts: `llama-3.3-70b-versatile` (change if your account uses another).

## License

Personal / portfolio project.
