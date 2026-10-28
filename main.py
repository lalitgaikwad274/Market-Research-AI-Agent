from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter

try:
    from langchain.agents import create_tool_calling_agent, AgentExecutor
except ImportError:
    from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from tools import scrape_tool, search_tool, save_to_txt

# Loads GOOGLE_API_KEY from .env
load_dotenv()


class LeadResponse(BaseModel):
    company: str
    contact_info: str
    email: str
    summary: str
    outreach_message: str
    tools_used: list[str]


class LeadResponseList(BaseModel):
    leads: list[LeadResponse]


import json
import re

# "gemini-2.5-flash" is the current active flash model
MODEL_NAME = "gemini-2.5-flash"

# Stay within free-tier limits (~5 requests/minute)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.07,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

from google.genai.types import HarmCategory, HarmBlockThreshold

# Ensure safety filters do not block legitimate business research queries
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_NONE,
}

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    rate_limiter=rate_limiter,
    max_retries=5,
    max_output_tokens=4096,
    temperature=0.2,
    safety_settings=safety_settings,
)

parser = PydanticOutputParser(pydantic_object=LeadResponseList)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a business research assistant helping to qualify small businesses in Vancouver, British Columbia that could benefit from IT services.

            Workflow instructions:
            1. Use the 'search' or 'scrape' tool to discover 5 local small businesses in Vancouver from various industries (e.g. dental/medical clinic, law office, accounting firm, local restaurant/retail, architectural studio).
            2. For each identified company, use the tools to research their operations and identify potential IT service needs (e.g. data backups, cybersecurity, cloud software, IT support).
            3. Once you have researched 5 companies, compile and return your final response as valid JSON matching the format below.

            Format Instructions:
            {format_instructions}
            """,
        ),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Saving is done in code below (reliable) instead of asking the LLM to call a save tool.
tools = [scrape_tool, search_tool]

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=20,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    stream_runnable=False,  # CRITICAL: prevents empty streaming chunks from triggering immediate premature AgentFinish
)


def output_to_text(output) -> str:
    """Extract text from output, properly handling thinking blocks and markdown fences."""
    if isinstance(output, str):
        text = output
    elif isinstance(output, list):
        text_parts = []
        for b in output:
            if isinstance(b, dict):
                # Only include actual text blocks, skip thinking blocks
                if b.get("type") == "text":
                    text_parts.append(b.get("text", ""))
                elif "text" in b and b.get("type") != "thinking":
                    text_parts.append(b["text"])
            else:
                text_parts.append(str(b))
        text = "".join(text_parts)
    else:
        text = str(output)

    # Strip markdown code fences if Gemini wraps JSON in ```json ... ```
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


if __name__ == "__main__":
    query = "Find and qualify exactly 5 local leads in Vancouver for IT Services. No more than 5 small businesses."

    raw_response = agent_executor.invoke({"query": query})
    raw_output = raw_response.get("output", "")
    cleaned_output = output_to_text(raw_output)

    if not cleaned_output:
        print("Error: The agent returned an empty output.")
        steps = raw_response.get("intermediate_steps", [])
        print(f"Total tool iterations executed: {len(steps)}")
    else:
        try:
            structured_response = parser.parse(cleaned_output)
            print("\n--- Qualified Leads ---")
            print(structured_response)
            save_msg = save_to_txt(structured_response.model_dump_json(indent=2))
            print(f"\n{save_msg}")
        except Exception as e:
            # Fallback to direct json loads + pydantic validation
            try:
                data = json.loads(cleaned_output)
                structured_response = LeadResponseList.model_validate(data)
                print("\n--- Qualified Leads (parsed via fallback) ---")
                print(structured_response)
                save_msg = save_to_txt(structured_response.model_dump_json(indent=2))
                print(f"\n{save_msg}")
            except Exception as e2:
                print("Error parsing response:", e)
                print("Raw Response Output:", repr(raw_output))