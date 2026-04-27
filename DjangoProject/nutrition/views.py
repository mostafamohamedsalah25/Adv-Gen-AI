from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import csv
import os

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, ToolMessage, RemoveMessage
from langchain.tools import tool
from langchain.agents.middleware import before_agent
from langgraph.runtime import Runtime
from pydantic import BaseModel
from tavily import TavilyClient

# Create your views here.


# ==========================================
# 1. Define Tools
# ==========================================

@tool
def search_nutrition_info(query: str, location: str = "") -> str:
    """Search the web for healthy restaurants, grocery stores, or nutrition information."""
    client = TavilyClient()
    search_query = f"{query} in {location}" if location else query
    result = client.search(search_query)
    return result


class MealRecord(BaseModel):
    user_name: str
    meal_description: str
    estimated_calories: int
    health_goals: str


@tool
def store_nutrition_data(record: MealRecord) -> str:
    """Store the user's meal analysis, estimated calories, and goals in a CSV file."""
    # Save the CSV in the root directory
    filename = 'nutrition_records.csv'
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['User Name', 'Meal Description', 'Estimated Calories', 'Health Goals'])
        writer.writerow([record.user_name, record.meal_description, record.estimated_calories, record.health_goals])

    return "Nutrition data stored successfully in CSV."


# ==========================================
# 2. Context Optimization (Middleware)
# ==========================================

@before_agent
def trim_messages(state: AgentState, runtime: Runtime) -> AgentState:
    """Trim tool messages and empty messages to avoid token limit issues."""
    trimed_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage) or msg.content == ""]
    return {"messages": [RemoveMessage(id=msg.id) for msg in trimed_messages]}


# ==========================================
# 3. Initialize Agent
# ==========================================

system_prompt = """You are a Multi-Modal Nutrition AI Assistant.
Your job is to analyze user dietary habits using text and images, estimate nutrition values, generate summaries, and provide recommendations.

TOOLS AT YOUR DISPOSAL:
1. `search_nutrition_info`: Use this to find nearby healthy food options or factual nutrition data.
2. `store_nutrition_data`: Use this to save a structured record of the user's meal when requested. ALWAYS ask for the user's name and health goals before storing.

STRICT CONSTRAINTS:
1. Do NOT provide medical diagnoses.
2. Do NOT provide strict diet plans.
3. CRITICAL: You MUST include the following disclaimer at the end of EVERY response that involves dietary advice: "This is not medical or dietary advice. Consult a qualified professional."
"""

llm = init_chat_model("gpt-4o-mini", temperature=0.5, max_tokens=1500)
memory = InMemorySaver()

nutrition_agent = create_agent(
    llm,
    system_prompt=system_prompt,
    tools=[search_nutrition_info, store_nutrition_data],
    checkpointer=memory,
    middleware=[trim_messages]
)


# ==========================================
# 4. Django View
# ==========================================

@csrf_exempt
def nutrition_interface(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            image_data = data.get('image', None)
            thread_id = data.get('thread_id', 'nutrition_session_default')

            config = {"configurable": {"thread_id": thread_id}}

            # Construct multimodal content
            message_content = []
            if user_message:
                message_content.append({"type": "text", "text": user_message})
            else:
                message_content.append({"type": "text", "text": "Can you analyze this meal for me?"})

            if image_data:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })

            # Invoke the Agent
            response = nutrition_agent.invoke({
                "messages": [HumanMessage(content=message_content)]
            }, config=config)

            ai_message = response['messages'][-1].content
            return JsonResponse({"reply": ai_message})

        except Exception as e:
            print(f"Backend Error: {e}")
            return JsonResponse({"reply": "Oops! Something went wrong on the server."}, status=500)

    # We will build this template in the next step!
    return render(request, 'nutrition/index.html')