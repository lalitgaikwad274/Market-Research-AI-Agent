# AI Agent with LangChain & Google Gemini

An intelligent lead qualification and market research agent built using **LangChain**, **Google Gemini (`gemini-2.5-flash`)**, **DuckDuckGo Search**, and **BeautifulSoup**.

The agent autonomously discovers local businesses, investigates their web footprint, qualifies their technology and IT requirements, and generates structured, production-ready lead data validated with **Pydantic**.

---

## Features

- **Autonomous Agentic Workflow**: Uses LangChain's Tool-Calling Agent architecture to iteratively search, scrape, evaluate, and synthesize information.
- **Multi-Tool Integration**:
  - `search`: Web search powered by DuckDuckGo with title, snippet, and URL extraction.
  - `scrape`: Custom HTML scraper using `BeautifulSoup` that parses clean text and handles direct URLs as well as search-and-scrape queries.
  - `save_to_text`: Structured persistent storage appending qualified leads to `leads_output.txt` with timestamps.
- **Strict Data Validation**: Validates all generated leads against Pydantic models (`LeadResponse`, `LeadResponseList`).
- **Free-Tier Rate Limiting**: Built-in `InMemoryRateLimiter` paces Gemini API requests to stay safely within Free Tier quotas (avoiding `429 RESOURCE_EXHAUSTED` errors).
- **Modern Compatibility**: Configured for LangChain 1.x / `langchain-classic` and Google GenAI SDK standards.

---

## Project Structure

```text
AI_Agent_with_Langchain/
├── main.py              # Agent definition, prompt template, model initialization & execution
├── tools.py             # Custom tools: DuckDuckGo search, web scraping, and file saving
├── requirements.txt     # Python package dependencies
├── .env                 # API keys and environment variables (ignored by Git)
├── .gitignore           # Git ignore configuration
├── leads_output.txt     # Generated output file containing qualified leads (created upon run)
└── README.md            # Project documentation
```

---

## Tech Stack

- **Python**: 3.10+ (tested on Python 3.12)
- **Framework**: LangChain (`langchain`, `langchain-classic`, `langchain-core`, `langchain-community`)
- **LLM Provider**: Google Generative AI (`langchain-google-genai`, `gemini-2.5-flash`)
- **Validation**: Pydantic v2
- **Web Search & Scraping**: `duckduckgo-search`, `requests`, `beautifulsoup4`
- **Environment**: `python-dotenv`

---

## Prerequisites

1. Python 3.10 or higher installed.
2. A **Google Gemini API Key**. You can obtain a free key from [Google AI Studio](https://aistudio.google.com/).

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AI_Agent_with_Langchain
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```bash
touch .env
```

Add your Gemini API key:
```env
GEMINI_API_KEY="your-gemini-api-key-here"
```
*(Alternatively, `GOOGLE_API_KEY="your-gemini-api-key-here"` is also supported.)*

---

## Usage

Run the agent script:

```bash
python3 main.py
```

### What Happens When You Run:
1. The agent initializes with the Gemini model and attached tools (`search`, `scrape`).
2. It launches an autonomous investigation to discover small businesses in the target location (default: Vancouver, BC).
3. For each business, it examines web pages to determine potential IT needs (backups, cybersecurity, managed services, cloud software).
4. Results are validated against the `LeadResponseList` Pydantic schema.
5. The final qualified leads are printed to the console and automatically saved to `leads_output.txt`.

---

## Data Schema

Each qualified lead contains:

| Field | Type | Description |
|---|---|---|
| `company` | `str` | Name of the business |
| `contact_info` | `str` | Phone, physical address, or website URL |
| `email` | `str` | Contact email address (or `"Not found"`) |
| `summary` | `str` | Assessment of their operations and potential IT requirements |
| `outreach_message`| `str` | Tailored cold email or outreach pitch |
| `tools_used` | `list[str]` | List of tools invoked to research this lead |

---

## Configuration & Tuning

### Changing the Target Query
In `main.py`, modify the `query` string inside the main execution block:
```python
query = "Find and qualify exactly 5 local leads in Toronto for Cloud Migration Services."
```

### Rate Limiter Tuning
In `main.py`, request pacing is handled by `InMemoryRateLimiter`:
```python
# gemini-2.5-flash Free Tier allows ~5 requests per minute
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.07,  # ~1 request every 14 seconds
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)
```
If you are on a paid Tier (Pay-As-You-Go), you can increase `requests_per_second` (e.g. `requests_per_second=1.0`) for faster execution.

---

## Troubleshooting

- **`ImportError: cannot import name 'create_tool_calling_agent'`**:
  In LangChain 1.x, classic agents are housed in `langchain-classic`. Ensure `langchain-classic` is installed (`pip install langchain-classic`).
- **`429 RESOURCE_EXHAUSTED`**:
  The Google AI Studio Free Tier has strict requests-per-minute limits. Do not remove the `InMemoryRateLimiter` configuration from `ChatGoogleGenerativeAI`.
- **`RuntimeWarning: duckduckgo_search has been renamed to ddgs`**:
  This warning is automatically suppressed in `tools.py`. If desired, you can also install `ddgs` directly via `pip install ddgs`.

