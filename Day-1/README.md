# Day 1 – Exploring LLM Behaviour and Building a Sweet Crumbs AI Assistant

## Objective
Understand the basic behaviour of Large Language Models (LLMs) through practical experiments and build a customized AI customer-support assistant for **Sweet Crumbs Home Bakery** using Groq and LangChain.

## Tasks
- Experiment with different temperature values (0.2 and 1.2) and compare AI responses.
- Explore hallucination by removing business facts from the system prompt.
- Understand conversation memory using separate calls and a combined messages list.
- Compare prompt token usage for the same question in English and Telugu.
- Customize the SYSTEM prompt for Sweet Crumbs using five business facts and three business rules.
- Test the customized assistant with three real customer questions.

## Implementation

### 1. Experiment with temperature

Ask the same question, "Write a tagline for Sweet Crumbs", three times at temperature `0.2` and three times at temperature `1.2`.

```python
from langchain_groq import ChatGroq

question = "Write a tagline for Sweet Crumbs"

llm_low = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)

llm_high = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=1.2
)

for i in range(3):
    print(llm_low.invoke(question).content)

for i in range(3):
    print(llm_high.invoke(question).content)
```

Low temperature generally produces more consistent responses, while high temperature encourages more creative and varied responses.

### 2. Explore hallucination

Temporarily remove the FACTS line from the system prompt and ask:

"Do you deliver to Kakinada?"

```python
SYSTEM_NO_FACTS = """
You are the assistant for Sweet Crumbs Home Bakery.

Rules:
- Never promise delivery times.
- If you do not know, say so.
"""

response = llm.invoke([
    ("system", SYSTEM_NO_FACTS),
    ("human", "Do you deliver to Kakinada?")
])

print(response.content)
```

This experiment helps identify whether the AI generates unsupported information when business facts are missing.

### 3. Understand conversation memory

First, make two separate calls.

```python
response1 = llm.invoke("My name is Ravi")
print(response1.content)

response2 = llm.invoke("What is my name?")
print(response2.content)
```

Then pass both messages together.

```python
messages = [
    ("system", SYSTEM),
    ("human", "My name is Ravi"),
    ("assistant", "Nice to meet you, Ravi!"),
    ("human", "What is my name?")
]

response = llm.invoke(messages)
print(response.content)
```

This demonstrates that separate calls do not automatically share conversation history. Passing previous messages allows the model to understand the conversation context.

### 4. Compare token usage in English and Telugu

Ask the same question in English and Telugu and compare their input token usage.

```python
english_question = "What cakes are available at Sweet Crumbs?"

telugu_question = "స్వీట్ క్రంబ్స్‌లో ఎలాంటి కేకులు అందుబాటులో ఉన్నాయి?"

english_response = llm.invoke(english_question)
telugu_response = llm.invoke(telugu_question)

print("English Prompt Tokens:",
      english_response.usage_metadata["input_tokens"])

print("Telugu Prompt Tokens:",
      telugu_response.usage_metadata["input_tokens"])
```

This experiment helps understand how tokenization differs between languages.

### 5. Customize the SYSTEM prompt for Sweet Crumbs

Create a business-specific system prompt with five facts and three rules.

**Business Facts:**
1. Sweet Crumbs is a home bakery.
2. Cakes are baked to order.
3. Orders require at least 24 hours' notice.
4. All cakes have eggless versions.
5. Customers can contact the bakery for cake order enquiries.

**Business Rules:**
1. Never invent cake prices or delivery availability.
2. Never promise delivery times without confirmation.
3. If you do not know something, honestly say so.

```python
SYSTEM = """
You are Sweet Crumbs AI, an assistant for Sweet Crumbs Home Bakery.

BUSINESS FACTS:
1. We are a home bakery.
2. We bake cakes to order.
3. Orders require at least 24 hours' notice.
4. All cakes have eggless versions.
5. Customers can contact us for cake order enquiries.

BUSINESS RULES:
1. Never invent cake prices or delivery availability.
2. Never promise delivery times without confirmation.
3. If you do not know something, honestly say so.

Reply politely and keep responses under 60 words.
"""

def ask_business(question):
    response = llm.invoke([
        ("system", SYSTEM),
        ("human", question)
    ])
    return response.content
```

### 6. Test with real customer questions

```python
print(ask_business("Can I order an eggless chocolate cake?"))

print(ask_business("Can I place an order for tomorrow?"))

print(ask_business(
    "Do you deliver to Kakinada and what is the delivery charge?"
))
```

These questions test whether the assistant follows the bakery's facts and rules when responding to customers.

## Result
Successfully explored five practical concepts of Large Language Models: temperature, hallucination, conversation memory, token usage, and business-specific system prompting.

A customized AI assistant for Sweet Crumbs Home Bakery was created to answer customer enquiries using predefined business facts and response rules.

## Notes
- Add `GROQ_API_KEY` to Google Colab Secrets before running the notebook.
- Temperature experiments use separate model instances.
- The hallucination experiment uses a temporary system prompt without business facts.
- Separate LLM calls do not automatically maintain conversation history.
- Token usage can differ between English and Telugu.
- The customized bakery assistant follows the five business facts and three business rules defined in its system prompt.
- Actual experiment outputs should be observed and compared after running the notebook.
