from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


load_dotenv()


class AgentState(TypedDict):
    question: str
    research: str
    review: str
    final_answer: str


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


def research_agent(state: AgentState):
    question = state["question"]

    prompt = f"""
You are the Researcher Agent.

Research and analyze the following question:

{question}

Provide clear and useful information.
Focus only on relevant facts.
"""

    response = llm.invoke(prompt)

    return {
        "research": response.content
    }


def review_agent(state: AgentState):
    research = state["research"]
    question = state["question"]

    prompt = f"""
You are the Reviewer Agent.

Review the research below and improve it.

Question:
{question}

Research:
{research}

Check the information for:
- relevance
- clarity
- missing important points
- unnecessary information

Then provide an improved version of the research.
"""

    response = llm.invoke(prompt)

    return {
        "review": response.content
    }


def final_agent(state: AgentState):
    question = state["question"]
    review = state["review"]

    prompt = f"""
You are the Final Answer Agent.

Create the final answer for the user's question using the reviewed research.

Question:
{question}

Reviewed Research:
{review}

Give a clear, concise and useful final answer.
"""

    response = llm.invoke(prompt)

    return {
        "final_answer": response.content
    }


graph = StateGraph(AgentState)


graph.add_node("researcher", research_agent)
graph.add_node("reviewer", review_agent)
graph.add_node("final_agent", final_agent)


graph.set_entry_point("researcher")

graph.add_edge("researcher", "reviewer")
graph.add_edge("reviewer", "final_agent")
graph.add_edge("final_agent", END)


app = graph.compile()


question = input("\nEnter your question: ")


result = app.invoke({
    "question": question,
    "research": "",
    "review": "",
    "final_answer": ""
})


print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(result["final_answer"])