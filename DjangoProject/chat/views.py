from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import List

# Create your views here.
class IngredientStatus(BaseModel):
    name: str
    status: str = Field(description="'available' or 'not available'")

class Meal(BaseModel):
    cooking_time: str
    number_of_individuals: int
    instructions: List[str]
    ingredients: List[IngredientStatus]

class ChefResponse(BaseModel):
    meals: List[Meal]


system_prompt = """
    You are an enthusiastic, professional AI Chef . 
    Your goal is to guide the user step-by-step to a meal decision based on the ingredients they provide.
    Strict Rules to Follow:
    1. Speak like a passionate chef.
    2. NEVER skip steps. You must ask questions to narrow down the meal choice.
    3. Understand what food is available.
    4. CRITICAL: When the user makes a final decision on the meal and you are ready to give the recipe, you MUST start your response with the exact keyword: [FINAL_RECIPE]
"""
    # 4. Once the user makes a final decision on a meal, you MUST output the final recipe using the exact structured JSON format requested.

llm = init_chat_model(
    "gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
)

structured_llm = llm.with_structured_output(ChefResponse)
memory = InMemorySaver()

# chef_agent = create_react_agent(
#     llm,
#     tools=[],
#     state_modifier=system_prompt,
#     checkpointer=memory
# )
chef_agent = create_agent(
    llm,
    tools=[], # No external tools needed for this specific lab
    system_prompt=system_prompt,
    checkpointer=memory
)

# 3. Define the View
@csrf_exempt
def chat_interface(request):
    if request.method == "POST":
        # Parse the incoming JSON data from the frontend
        data = json.loads(request.body)
        user_message = data.get('message', '')
        thread_id = data.get('thread_id', 'default_session')

        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the LangGraph agent
        response = chef_agent.invoke({
            "messages": [HumanMessage(content=user_message)]
        }, config=config)

        ai_message = response['messages'][-1].content

        if "[FINAL_RECIPE]" in ai_message:
            # Clean the keyword from the message sent to the user
            clean_message = ai_message.replace("[FINAL_RECIPE]", "").strip()

            # Pass the entire conversation history to the structured LLM
            # so it knows exactly what meal was decided and what ingredients exist
            conversation_history = response['messages']
            structured_res = structured_llm.invoke(conversation_history)

            return JsonResponse({
                "reply": clean_message,
                "recipe_json": structured_res.model_dump()  # Convert Pydantic to dict
            })

        # If the AI implies the meal is decided, we could trigger the structured output here.
        # For now, we return the conversational response to the frontend.
        return JsonResponse({"reply": ai_message})

    # If GET request, render the HTML template (we will build this next)
    return render(request, 'chat/index.html')