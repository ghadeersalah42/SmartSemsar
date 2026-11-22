"""Structured output من Groq مع fallback.

gpt-oss-20b على Groq أحياناً بيرجع رد فاضي بدل الـ tool call، فـ Groq بيرمي:
    Error code: 400 - tool_use_failed ("Tool choice is required, but model did not call a tool")
فلو الطريقة العادية (function calling) وقعت، بنطلب نفس الـ schema بطريقة json_schema (من غير tools)،
ولو دي كمان وقعت بنجرب function calling مرة كمان.
"""
from langchain_groq import ChatGroq


def structured_llm(schema, model_name: str = "openai/gpt-oss-20b"):
    llm = ChatGroq(model_name=model_name, temperature=0)
    tools = llm.with_structured_output(schema)
    json_schema = llm.with_structured_output(schema, method="json_schema")
    return tools.with_fallbacks([json_schema, tools])
