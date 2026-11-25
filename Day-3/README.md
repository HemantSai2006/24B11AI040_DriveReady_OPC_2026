# Day 3 – Building and Testing a Business AI Agent

## Objective

To build a business-specific AI assistant using LangChain and Groq. The main objective is to create a bakery chatbot, maintain conversation history, test its responses using CSV test cases, improve the system prompt, experiment with temperature settings, and understand the difference between plain code, LLM calls, and AI agents.

## Business Details

**Business Name:** Sweet Crumbs Home Bakery

Business information used in the chatbot:

- Plum Cake – ₹500
- Chocolate Cake – ₹650
- Cupcake – ₹80 each
- Orders require 24 hours' advance notice.
- All cakes have eggless versions.
- Delivery time and location must be confirmed with the owner.

## Implementation

### 1. Installing Required Libraries

Installed the required Python packages.

```python
!pip install -q langchain langchain-groq
```

### 2. Configuring Groq API

Configured the Groq API key using Google Colab Secrets and initialized the language model.

```python
import os
from google.colab import userdata

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3
)
```

### 3. Creating the Business SYSTEM Prompt

Created a SYSTEM prompt containing the bakery's role, menu, business policies, rules, and response format.

```python
SYSTEM = """ROLE: You are the friendly assistant for Sweet Crumbs Home Bakery.

FACTS: Plum cake Rs 500. Chocolate cake Rs 650. Cupcake Rs 80 each.
Orders need 24 hours' notice. All cakes have eggless versions.

RULES: Never promise a delivery time or place.
If it isn't in FACTS, say "I'll check with the owner."
Work out totals carefully.

FORMAT: Under 50 words, friendly,
English or Telugu matching the customer."""
```

### 4. Creating the Chatbot Using LangChain

Used `ChatPromptTemplate`, `MessagesPlaceholder`, and `StrOutputParser` to create the chatbot chain.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()
```

### 5. Implementing Conversation Memory

Created a history list to remember previous customer messages and responses.

```python
history = []

def ask(question: str) -> str:
    answer = chain.invoke({
        "history": history,
        "question": question
    })

    history.extend([
        ("human", question),
        ("ai", answer)
    ])

    return answer
```

Tested the chatbot with customer questions to check whether it remembers the customer's name.

### 6. Creating CSV-Based Test Cases

Created a `test.csv` file containing 10 customer questions and their expected keywords.

The CSV contains two columns:

- `question` – Customer question.
- `must_include` – Expected keyword in the answer.

Example:

```csv
question,must_include
"What is the price of a plum cake?","500"
"Do you make eggless cakes?","eggless"
"How much advance notice is required?","24 hours"
```

Uploaded the CSV file to Google Colab and saved it at:

`/content/test.csv`

### 7. Running Automated Tests

Used Python's CSV module to read test cases and check whether the chatbot responses contain the expected keywords.

```python
import csv

def run_tests(path: str = "/content/test.csv") -> None:
    rows = list(csv.DictReader(open(path)))
    passed = 0

    for row in rows:
        answer = chain.invoke({
            "history": [],
            "question": row["question"]
        })

        ok = row["must_include"].lower() in answer.lower()
        passed += ok

        print(
            "PASS" if ok else "FAIL",
            "|",
            row["question"],
            "→",
            answer[:70]
        )

    print(f"\n{passed}/{len(rows)} passed ({passed / len(rows):.0%})")
```

The test function calculates the pass rate based on the number of successful test cases.

### 8. Improving the SYSTEM Prompt

Improved the prompt by adding a rule to calculate the correct total price when customers order multiple items.

New instruction:

```text
When customers order multiple items,
calculate each item's price and provide
the correct final total in rupees.
```

Ran the same CSV test cases again and compared the pass rate before and after the improvement.

### 9. Experimenting with Temperature

Changed the model temperature from `0.3` to `1.0` and ran the same test cases twice.

Compared the responses and pass rates to observe whether the chatbot generated different answers between runs.

### 10. Creating an Agent Using LangChain

Used LangChain's `create_agent` function to create a basic bakery agent.

```python
from langchain.agents import create_agent

agent_order = create_agent(
    model=llm,
    tools=[],
    system_prompt=SYSTEM
)

result_order = agent_order.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Do you make eggless cakes?"
        }
    ]
})

print(result_order["messages"][-1].content)
```

Created the agent without tools as an initial experiment. Tools can be added to perform business tasks in future exercises.

## 11. Agent or Not – Business Tasks

| Business Job | Category | Reason |
|---|---|---|
| Calculate the total price of bakery items | Plain Code | Uses mathematical calculations and fixed menu prices. |
| Answer customer questions about bakery products | One LLM Call | Understands customer questions and generates responses. |
| Handle an order by checking stock, calculating the bill, and arranging delivery | Agent | Requires multiple steps and tools to complete the task. |

## Result

- Successfully created a Sweet Crumbs Home Bakery AI assistant using LangChain and Groq.
- Implemented a SYSTEM prompt with business-specific facts and rules.
- Implemented conversation history to remember previous messages.
- Created CSV-based test cases for evaluating chatbot responses.
- Tested prompt improvements and different temperature settings.
- Created a basic LangChain agent using `create_agent`.
- Understood the differences between plain Python code, one LLM call, and an AI agent.

## Notes

- The Groq API key was configured using Google Colab Secrets.
- The model used was `openai/gpt-oss-120b`.
- The initial temperature was `0.3`; it was changed to `1.0` for experimentation.
- The CSV test file was stored at `/content/test.csv`.
- The test pass rate should be recorded from the actual Colab execution results.
- The agent was initially created with an empty tools list (`tools=[]`).
