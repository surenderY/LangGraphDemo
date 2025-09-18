# LangGraphDemo

## Overview
	This Python script implements a conversational AI chatbot using LangGraph and LangChain frameworks. The chatbot integrates with a language model (via OpenAI) 
	and has the ability to use external tools (specifically Tavily Search) to enhance its responses with real-time information.

### Key Components Dependencies
	LangChain: For chat model initialization and tool integration
	LangGraph: For creating a directed graph that manages conversation flow
	Tavily Search: External search tool for retrieving information
	OpenAI: Underlying LLM provider

### Architecture
	The chatbot is built using a graph-based architecture with the following components:

	State Management:

		- Uses a TypedDict class to maintain conversation state
		- Stores message history with annotations for proper message handling
	
	Graph Structure:

		- Two main nodes: "chatbot" and "tools"
		- Conditional routing between nodes based on tool usage requirements
		- Defined flow from START → chatbot → (tools) → chatbot → END

        ![HCOrderStatus Agent LangGraph](/data/hcorderstatus.png "HCOrderStatus Agent LangGraph")
	
	Tool Integration:

		- Implements TavilySearch for web search capabilities
		- Uses a BasicToolNode class to handle tool execution
		- Supports dynamic tool routing based on LLM decisions