# Day 2 – Building a Conversational Bakery Assistant with Groq and Python

## Objective
Build a conversational AI assistant for **Sweet Crumbs Home Bakery** using the Groq API and Python. Practise Python data handling, menu price calculation, function decorators, conversation history, and basic API error handling.

## Tasks
- Install and import the required Groq and LangChain packages.
- Configure the Groq API key using Google Colab Secrets.
- Initialize the Groq client.
- Define the bakery menu, business facts, and chatbot instructions.
- Create a Python decorator to log function execution.
- Implement a reusable function to calculate bakery item prices.
- Maintain recent conversation history using a message list.
- Build a command-line chatbot with basic API error handling.

## Implementation

### 1. Install and import the required packages

```python
!pip install langchain_groq
!pip install -q langchain_groq groq

from langchain_groq import ChatGroq
import os
import json
import functools
from groq import Groq
from openai import RateLimitError, APIConnectionError
```

The required packages are installed and imported to connect to Groq, handle Python data, and implement the chatbot.

### 2. Configure the Groq API key and initialize the client

```python
from google.colab import userdata

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

print("Groq client initialized successfully!")
```

The API key is retrieved securely from Google Colab Secrets instead of being directly written in the notebook.

The Groq client is initialized using the retrieved API key to send chat-completion requests.

### 3. Define the model, menu, and system prompt

```python
MODEL = "openai/gpt-oss-20b"

MENU = {
    "plum cake": 500,
    "chocolate cake": 650,
    "cupcake": 80
}
```

A Python dictionary stores the bakery menu and item prices in rupees.

The system prompt defines the assistant's role and business information.

**Business Facts:**
- Sweet Crumbs is a home bakery.
- Plum cake costs ₹500.
- Chocolate cake costs ₹650.
- Cupcake costs ₹80.
- Orders require 24 hours' notice.
- All cakes have eggless versions.

**Business Rules:**
- Never promise a delivery time.
- If information is unavailable, say "I'll check with the owner."
- Remember the customer's name and order during the conversation.

**Response Format:**
- Keep responses under 50 words.
- Reply in a friendly manner.
- Respond in English or Telugu according to the customer's language.

### 4. Create a logging decorator

```python
def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"   [log] {func.__name__}{args} -> {result!r}")
        return result
    return wrapper
```

The `log_call` decorator is used to display the function name, arguments, and returned result whenever the decorated function is executed.

It helps track function execution and understand how the program works.

### 5. Create a reusable price calculation function

```python
@log_call
def get_price(item: str, qty: int = 1) -> int:
    """Return the total price in rupees for qty of an item on the menu."""
    return MENU[item.lower().strip()] * qty
```

The `get_price()` function calculates the total price of a bakery item based on its quantity.

It uses the menu dictionary to retrieve the item price and multiplies it by the requested quantity.

For example:

```python
print("Total for 6 cupcakes:", get_price("cupcake", 6))
```

**Expected Output:**

```text
[log] get_price('cupcake', 6) -> 480
Total for 6 cupcakes: 480
```

### 6. Maintain conversation history

```python
history = [
    {"role": "system", "content": SYSTEM}
]

MAX_TURNS = 8
```

A list named `history` stores the system instructions and previous conversation messages.

The `MAX_TURNS` variable limits the number of recent message entries sent to the model.

```python
recent = [history[0]] + history[1:][-MAX_TURNS:]
```

This keeps the system message along with the latest eight conversation messages in each API request.

### 7. Implement the chatbot function

```python
def chat(user_text: str) -> str:
    history.append({
        "role": "user",
        "content": user_text
    })

    recent = [history[0]] + history[1:][-MAX_TURNS:]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=recent,
            temperature=0.3
        )

        reply = response.choices[0].message.content

    except RateLimitError:
        reply = "We're a little busy. Please try again in a minute."

    except APIConnectionError:
        reply = "I can't reach the server right now. Please try again shortly."

    history.append({
        "role": "assistant",
        "content": reply
    })

    return reply
```

The `chat()` function accepts a customer message and sends it along with recent conversation history to the Groq API.

The assistant's response is stored in the history list so that the chatbot can maintain conversation context.

It also handles two common API errors:

- `RateLimitError`: Handles API rate-limit issues.
- `APIConnectionError`: Handles connection problems.

### 8. Build the command-line chatbot

```python
print("Total for 6 cupcakes:", get_price("cupcake", 6))

while True:
    text = input("You: ")

    if text.lower().strip() in ("bye", "exit", "quit"):
        print("Bot: Thank you for visiting Sweet Crumbs!")
        break

    print("Bot:", chat(text))
```

A continuous input loop allows customers to interact with the bakery assistant.

The chatbot continues responding until the customer enters `bye`, `exit`, or `quit`.

## Result
Successfully created a conversational AI assistant for Sweet Crumbs Home Bakery using Python and the Groq API.

The implementation demonstrates menu price calculation, function decorators, conversation history, and basic API error handling. The chatbot can respond to customer enquiries while using the predefined bakery information and instructions.

## Notes
- Add `GROQ_API_KEY` to Google Colab Secrets before running the notebook.
- The chatbot uses the `openai/gpt-oss-20b` model through the Groq client.
- The menu is stored as a Python dictionary.
- The `get_price()` function calculates item totals and uses a decorator to log execution.
- `MAX_TURNS = 8` retains the latest eight individual conversation messages in the request context, in addition to the system message.
- Conversation history is maintained only during the current Python session.
- The chatbot does not save conversations permanently to a database or file.
- The price calculation function is separate from the chatbot; the model does not automatically execute it when customers ask about prices.
- API errors are handled using `RateLimitError` and `APIConnectionError`.
- The notebook imports `ChatGroq`, but the final chatbot uses the Groq Python SDK directly.


