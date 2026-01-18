"""Agentic workflow using LangGraph for legal Q&A with tools."""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from utils.agents.prompts import SYSTEM_PROMPT
from utils.agents.tools.date_caculator import calculate_date_difference
from rag.vector_db import QdrantDB
from utils.chat_client import ChatClient

class AgentState(TypedDict):
    """State for the agentic workflow."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class AgenticWorkflow:
    """Single-node agentic workflow with tool usage."""
    
    def __init__(
        self,
        llm : ChatClient,
        vector_db : QdrantDB
    ):
        """Initialize the agentic workflow.
        
        Args:
            llm: LangChain LLM instance (from ChatClient).
        """
        self.llm = llm
        
        # Create tools - only date calculator for now
        tools = [calculate_date_difference]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.tools_dict = {tool.name: tool for tool in tools}
        
        # Build the graph with single agent node
        self.graph = self._build_graph()
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """Single agent node that handles everything - LLM calls and tool execution."""
        messages = state["messages"]
        
        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        
        # Keep looping until we get a final response (no tool calls)
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            # Call LLM
            response = self.llm_with_tools.invoke(messages)
            
            # Check if tool calls are needed
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Execute tools
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    if tool_name in self.tools_dict:
                        tool = self.tools_dict[tool_name]
                        try:
                            tool_result = tool.invoke(tool_args)
                            tool_messages.append(
                                ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_id
                                )
                            )
                        except Exception as e:
                            tool_messages.append(
                                ToolMessage(
                                    content=f"Error executing tool: {str(e)}",
                                    tool_call_id=tool_id
                                )
                            )
                    else:
                        tool_messages.append(
                            ToolMessage(
                                content=f"Tool {tool_name} not found",
                                tool_call_id=tool_id
                            )
                        )
                
                # Add response and tool messages to conversation
                messages = messages + [response] + tool_messages
                iteration += 1
            else:
                # No tool calls, we're done
                return {"messages": [response]}
        
        # Max iterations reached, return last response
        return {"messages": [response]}
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with a single agent node."""
        workflow = StateGraph(AgentState)
        
        # Add single agent node
        workflow.add_node("agent", self._agent_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Single node goes directly to END
        workflow.add_edge("agent", END)
        
        return workflow.compile()
    
    def invoke(self, user_message: str, chat_history: list = None) -> str:
        """Invoke the agentic workflow with a user message.
        
        Args:
            user_message: The user's question or message.
            chat_history: Optional list of previous messages for context.
            
        Returns:
            The final response from the agent.
        """
        # Build initial messages
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append(HumanMessage(content=user_message))
        
        # Run the workflow
        final_state = self.graph.invoke({"messages": messages})
        
        # Extract the final response
        final_messages = final_state["messages"]
        # Get the last AI message
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        
        return "I apologize, but I couldn't generate a response."
    
    def stream(self, user_message: str, chat_history: list = None):
        """Stream the agentic workflow response.
        
        Args:
            user_message: The user's question or message.
            chat_history: Optional list of previous messages for context.
            
        Yields:
            Response chunks as they are generated (AIMessage objects with content).
        """
        # Build initial messages
        messages = []
        if chat_history:
            messages.extend(chat_history)
        messages.append(HumanMessage(content=user_message))
        
        # Stream the workflow execution
        last_ai_message = None
        for chunk in self.graph.stream({"messages": messages}):
            if "agent" in chunk:
                agent_messages = chunk["agent"]["messages"]
                for msg in agent_messages:
                    if isinstance(msg, AIMessage):
                        if msg.content:
                            # If this is a new message or has more content, yield it
                            if last_ai_message is None or msg.content != last_ai_message.content:
                                last_ai_message = msg
                                yield msg
        
        # If we didn't get a final response, get it from the final state
        if last_ai_message is None or not last_ai_message.content:
            final_state = self.graph.invoke({"messages": messages})
            final_messages = final_state["messages"]
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    yield msg
                    break
