from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import operator
import uuid
import functools
import os

from OrdersStatusChat import OrdersAgent, get_order_details, update_quantity, system_prompt


class RouterAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class RouterAgent:

    def __init__(self, model, system_prompt, smalltalk_prompt, debug=False):
        
        self.system_prompt=system_prompt
        self.smalltalk_prompt=smalltalk_prompt
        self.model=model
        self.debug=debug
        
        router_graph=StateGraph(RouterAgentState)
        router_graph.add_node("Router",self.call_llm)
        router_graph.add_node("Orders_Agent",orders_node)
        router_graph.add_node("Small_Talk", self.respond_smalltalk)
                              
        router_graph.add_conditional_edges(
            "Router",
            self.find_route,
            {"ORDER" : "Orders_Agent",
             "SMALLTALK" : "Small_Talk",
             "END": END }
        )

        #One way routing, not coming back to router
        router_graph.add_edge("Orders_Agent",END)
        router_graph.add_edge("Small_Talk",END)
        
        #Set where there graph starts
        router_graph.set_entry_point("Router")
        self.router_graph = router_graph.compile()

    def call_llm(self, state:RouterAgentState):
        messages=state["messages"]
        if self.debug:
            print(f"Call LLM received {messages}")
            
        #If system prompt exists, add to messages in the front
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages

        #invoke the model with the message history
        result = self.model.invoke(messages)

        if self.debug:
            print(f"Call LLM result {result}")
        return { "messages":[result] }

    def respond_smalltalk(self, state:RouterAgentState):
        messages=state["messages"]
        if self.debug:
            print(f"Small talk received: {messages}")
            
        #If system prompt exists, add to messages in the front
        
        messages = [SystemMessage(content=self.smalltalk_prompt)] + messages

        #invoke the model with the message history
        result = self.model.invoke(messages)

        if self.debug:
            print(f"Small talk result {result}")
        return { "messages":[result] }
        
    def find_route(self, state:RouterAgentState):
        last_message = state["messages"][-1]
        if self.debug: 
            print("Router: Last result from LLM : ", last_message)

        #Set the last message as the destination
        destination=last_message.content

        if self.debug:
            print(f"Destination chosen : {destination}")
        return destination
    
# Helper function to invoke an agent
def agent_node(state, agent, name, config):

    #extract thread-id from request for conversation memory
    thread_id=config["metadata"]["thread_id"]
    #Set the config for calling the agent
    agent_config = {"configurable": {"thread_id": thread_id}}

    #Pass the thread-id to establish memory for chatbot
    #Invoke the agent with the state
    result = agent.invoke(state, agent_config)

    # Convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        final_result=AIMessage(result['messages'][-1].content)
    return {
        "messages": [final_result]
    }


if __name__ == "__main__":

    print("\n#####################################################")
    print("     Welcome to HC Order status chatbot application")
    print("#####################################################\n")
    load_dotenv()
    llm_model = os.getenv("LLM_MODEL")
    model = init_chat_model(llm_model)

    orders_agent = OrdersAgent(model, 
                           [get_order_details, update_quantity], 
                           system_prompt,
                           debug=True)
    
    #Create the Orders node
    #For a custom agent, the agent graph need to be provided as input
    orders_node=functools.partial(agent_node,
                                agent=orders_agent.agent_graph,
                                name="Orders_Agent")

    #Setup the system problem
    router_prompt = """ 
        You are a Router, that analyzes the input query and chooses 4 options:
        SMALLTALK: If the user input is small talk, like greetings and good byes.
        ORDER: If the query is about orders for laptops, like order status, order details or update order quantity
        END: Default, its ORDER.

        The output should only be just one word out of the possible 4 : SMALLTALK, ORDER, END.
        """

    smalltalk_prompt = """
        If the user request is small talk, like weather, greetings and goodbyes, respond professionally.
        Mention that you will be able to answer questions about laptop product features and provide order status and updates.
        """

    router_agent = RouterAgent(model, 
                            router_prompt, 
                            smalltalk_prompt,
                            debug=True)
    
    # Image(router_agent.router_graph.get_graph().draw_mermaid_png())

    #Create a new thread
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    #Execute a single request
    # "Tell me about the features of SpectraBook"c
    # messages=[HumanMessage(content="How are you doing?")]
    # result = router_agent.router_graph.invoke({"messages":messages}, config)
    # for message in result['messages']:
    #     print(message.pretty_repr())

    user_inputs = [
        # "How are you doing?",
        # "what is the weather in NOVI, Michigan?",
        "Please show me the details of the order ORD-7311",
        # "Can you add one more of that laptop to the order? ",
        # "Please show me the details of the order ORD-7311",
        # "Bye"
    ]

    for input in user_inputs:
        print(f"----------------------------------------\nUSER : {input}")
        #Format the user message
        user_message = {"messages":[HumanMessage(input)]}
        #Get response from the agent
        ai_response = router_agent.router_graph.invoke(user_message,config=config)
        #Print the response
        print(f"\nAGENT : {ai_response['messages'][-1].content}")