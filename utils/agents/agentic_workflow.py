"""Agentic workflow using LangGraph for legal Q&A with tools."""

from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langfuse.langchain import CallbackHandler
from langfuse import observe

from utils.agents.prompts import (
    RELEVANCY_CHECKER_PROMPT,
    TASK_PLANNER_PROMPT,
    TASK_EXECUTOR_PROMPT,
    RAG_REPHRASER_PROMPT,
    RAG_RELEVANCE_CHECKER_PROMPT,
    ANSWER_SUMMARIZER_PROMPT,
    ANSWER_COMPLETENESS_PROMPT,
    GENERIC_RESPONSE
)
from utils.agents.tools.date_caculator import calculate_date_difference
from utils.agents.tools.rag_retrieval import RAGRetrieval
from utils.rag.vector_db import QdrantDB
from utils.chat_client import ChatClient


# ==================== TYPE DEFINITIONS ====================
class SubTask(TypedDict):
    """A subtask to be executed."""
    question: str
    answer: Optional[str]
    status: str  # "pending", "completed", "failed"


class RAGState(TypedDict):
    """State for RAG subgraph."""
    original_query: str
    rephrased_queries: List[str]
    retrieved_chunks: List[str]
    relevant_chunks: List[str]
    retry_count: int
    is_relevant: bool


class AgentState(TypedDict):
    """State for the agentic workflow."""
    messages: List[BaseMessage]
    chat_history: List[BaseMessage]  # Previous conversation history
    original_question: str
    is_relevant: bool
    subtasks: List[SubTask]
    current_task_idx: int
    rag_state: Optional[RAGState]
    final_answer: str
    replanning_count: int
    needs_split: bool
    task_to_split: Optional[str]


# ==================== HELPER FUNCTIONS ====================
def check_relevance_keyword(text: str) -> bool:
    """Check if response indicates relevance (RELEVANT but not NOT_RELEVANT)."""
    upper = text.upper()
    return "RELEVANT" in upper and "NOT_RELEVANT" not in upper


def create_rag_state(query: str, retry_count: int = 0, **overrides) -> RAGState:
    """Create a new RAG state with defaults."""
    return {
        "original_query": query,
        "rephrased_queries": [],
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "retry_count": retry_count,
        "is_relevant": False,
        **overrides
    }


def create_initial_state(user_message: str, chat_history: List[BaseMessage] = None) -> AgentState:
    """Create the initial workflow state."""
    return {
        "messages": [HumanMessage(content=user_message)],
        "chat_history": chat_history or [],
        "original_question": user_message,
        "is_relevant": False,
        "subtasks": [],
        "current_task_idx": 0,
        "rag_state": None,
        "final_answer": "",
        "replanning_count": 0,
        "needs_split": False,
        "task_to_split": None
    }


# ==================== NODE DISPLAY NAMES ====================
NODE_DISPLAY_NAMES = {
    "relevancy_checker": "🔍 Checking Relevancy",
    "generic_response": "💬 Generating Response",
    "task_planner": "📋 Planning Tasks",
    "task_executor": "⚙️ Executing Task",
    "subtask_summarizer": "📝 Summarizing Subtask",
    "check_more_tasks": "✅ Checking Progress",
    "rag_rephraser": "✏️ Rephrasing Query",
    "rag_retrieval_and_relevance_checker": "📚 Searching Documents",
    "answer_summarizer": "📝 Summarizing Answer",
    "completeness_check": "🔎 Verifying Completeness",
    "replan": "🔄 Replanning",
}


class AgenticWorkflow:
    """Multi-node agentic workflow with task planning and execution."""
    
    MAX_RAG_RETRIES = 3
    MAX_REPLANS = 1
    
    def __init__(self, llm: ChatClient, vector_db: QdrantDB, max_task_count: int = 3):
        self.llm = llm
        self.max_task_count = max_task_count
        self.rag_retrieval = RAGRetrieval(vector_db)
        self.search_tool = self.rag_retrieval.tool
        self.date_tool = calculate_date_difference
        self.langfuse_handler = CallbackHandler()
        self.graph = self._build_graph()
    
    # ==================== LLM HELPERS ====================
    def _format_chat_history(self, chat_history: List[BaseMessage]) -> str:
        """Format chat history as a string for inclusion in prompts.
        
        Note: The history is already limited by the app layer via CHAT_HISTORY_LIMIT env var.
        """
        if not chat_history:
            return ""
        
        formatted = []
        for msg in chat_history:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            # Truncate long messages to keep context manageable
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def _ask_llm(self, system_prompt: str, user_content: str, chat_history: List[BaseMessage] = None) -> str:
        """Send a message to LLM and return the response content.
        
        Note: Chat history is already limited by the app layer via CHAT_HISTORY_LIMIT env var.
        """
        messages = [SystemMessage(content=system_prompt)]
        
        # Include chat history if provided (for context-aware responses)
        if chat_history:
            messages.extend(chat_history)
        
        messages.append(HumanMessage(content=user_content))
        response = self.llm.invoke(messages)
        return response.content
    
    # ==================== NODE 1: RELEVANCY CHECKER ====================
    def _relevancy_checker(self, state: AgentState) -> AgentState:
        """Check if question is relevant to our capabilities."""
        chat_history = state.get("chat_history", [])
        
        # Include chat history context if available
        user_content = f"Question: {state['original_question']}"
        if chat_history:
            history_context = self._format_chat_history(chat_history)
            user_content = f"Previous conversation:\n{history_context}\n\nCurrent question: {state['original_question']}"
        
        response = self._ask_llm(
            RELEVANCY_CHECKER_PROMPT,
            user_content
        )
        return {**state, "is_relevant": check_relevance_keyword(response)}
    
    def _route_relevancy(self, state: AgentState) -> str:
        """Route based on relevancy check."""
        return "task_planner" if state["is_relevant"] else "generic_response"
    
    def _generic_response(self, state: AgentState) -> AgentState:
        """Return generic capability response."""
        return {**state, "final_answer": GENERIC_RESPONSE}
    
    # ==================== NODE 2: TASK PLANNER ====================
    def _task_planner(self, state: AgentState) -> AgentState:
        """Split question into subtasks."""
        task_to_split = state.get("task_to_split")
        existing_subtasks = list(state.get("subtasks", []))
        current_idx = state.get("current_task_idx", 0)
        chat_history = state.get("chat_history", [])
        
        # Determine what question to plan
        if task_to_split:
            question_to_plan = task_to_split
        else:
            question_to_plan = state["original_question"]
            failed = [t["question"] for t in existing_subtasks if t["status"] == "failed"]
            if failed:
                question_to_plan += "\n\nPreviously failed:\n" + "\n".join(failed)
        
        # Build user content with chat history context
        user_content = f"Question: {question_to_plan}"
        if chat_history and not task_to_split:
            history_context = self._format_chat_history(chat_history)
            user_content = f"Previous conversation (for context):\n{history_context}\n\nCurrent question to plan: {question_to_plan}"
        
        # Get subtasks from LLM
        prompt = TASK_PLANNER_PROMPT.format(max_task_count=self.max_task_count)
        response = self._ask_llm(prompt, user_content)
        
        # Parse subtasks from numbered list
        new_subtasks = self._parse_subtasks(response, question_to_plan)
        
        # Merge subtasks if re-splitting
        if task_to_split and existing_subtasks:
            final_subtasks = existing_subtasks[:current_idx] + new_subtasks + existing_subtasks[current_idx + 1:]
        else:
            final_subtasks = new_subtasks
        
        return {
            **state,
            "subtasks": final_subtasks,
            "current_task_idx": current_idx,
            "needs_split": False,
            "task_to_split": None
        }
    
    def _parse_subtasks(self, response: str, fallback_question: str) -> List[SubTask]:
        """Parse subtasks from LLM response."""
        subtasks = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                task_text = line.lstrip("0123456789.-) ").strip()
                if task_text:
                    subtasks.append(SubTask(question=task_text, answer=None, status="pending"))
        
        return subtasks or [SubTask(question=fallback_question, answer=None, status="pending")]
    
    # ==================== NODE 3: TASK EXECUTOR ====================
    def _task_executor(self, state: AgentState) -> AgentState:
        """Execute current subtask using tool calling."""
        subtasks = list(state["subtasks"])
        idx = state["current_task_idx"]
        
        if idx >= len(subtasks):
            return state
        
        question = subtasks[idx]["question"]
        
        # Use tools already initialized in __init__
        tools = [self.date_tool, self.search_tool]
        
        # Let LLM decide which tool to use
        llm_with_tools = self.llm.bind_tools(tools)
        response = llm_with_tools.invoke([
            SystemMessage(content=TASK_EXECUTOR_PROMPT),
            HumanMessage(content=f"Task: {question}")
        ])
        
        # Get tool calls from response
        tool_calls = response.tool_calls if response.tool_calls else []
        
        # Process all tool calls - collect date results and check for RAG trigger
        date_results = []
        trigger_rag = False
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            if tool_name == "calculate_date_difference":
                # Execute date calculation and collect result
                result = self.date_tool.invoke(tool_args)
                date_results.append(result)
            
            elif tool_name == "search_legal_documents":
                trigger_rag = True
        
        # Build enriched question with date calculation results if any
        enriched_question = question
        if date_results:
            enriched_question = f"{question}\n\nDate calculation results:\n" + "\n".join(date_results)
        
        # Route based on tool calls
        if trigger_rag:
            # Trigger RAG subgraph with enriched question
            return {
                **state,
                "rag_state": create_rag_state(enriched_question),
                "needs_split": False
            }
        
        if date_results:
            # Only date calculation needed - mark as completed
            subtasks[idx] = SubTask(question=question, answer="\n".join(date_results), status="completed")
            return {**state, "subtasks": subtasks, "needs_split": False}
        
        # No tool called - use LLM response directly (shouldn't happen for legal questions)
        answer = response.content if response.content else "No answer generated."
        subtasks[idx] = SubTask(question=question, answer=answer, status="completed")
        return {**state, "subtasks": subtasks, "needs_split": False}
    
    def _route_after_executor(self, state: AgentState) -> str:
        """Route after task execution."""
        if state.get("needs_split"):
            return "task_planner"
        rag_state = state.get("rag_state")
        if rag_state and not rag_state.get("is_relevant", False):
            return "rag_rephraser"
        return "subtask_summarizer"
    
    def _subtask_summarizer(self, state: AgentState) -> AgentState:
        """Summarize the current subtask answer using the same format as the final answer summarizer."""
        subtasks = list(state["subtasks"])
        idx = state["current_task_idx"]
        chat_history = state.get("chat_history", [])
        
        if idx >= len(subtasks) or not subtasks[idx].get("answer"):
            return state
        
        current_task = subtasks[idx]
        
        # Build user content with chat history for context-aware summarization
        user_content = f"Original Question: {current_task['question']}\n\nSubtask Answers:\n{current_task['answer']}"
        if chat_history:
            history_context = self._format_chat_history(chat_history)
            user_content = f"Previous conversation (for context - helps understand follow-up questions):\n{history_context}\n\n{user_content}"
        
        # Use tool calling to allow date calculations during summarization
        llm_with_tools = self.llm.bind_tools([self.date_tool])
        response = llm_with_tools.invoke([
            SystemMessage(content=ANSWER_SUMMARIZER_PROMPT),
            HumanMessage(content=user_content)
        ])
        
        # Process any tool calls (date calculations)
        summarized_answer = response.content or ""
        if response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                if tool_call["name"] == "calculate_date_difference":
                    result = self.date_tool.invoke(tool_call["args"])
                    tool_results.append(result)
            
            # If there were tool calls, ask LLM to incorporate results
            if tool_results:
                followup_content = f"Original response: {summarized_answer}\n\nDate calculation results:\n{chr(10).join(tool_results)}\n\nPlease incorporate these date calculations into your final response."
                final_response = self._ask_llm(ANSWER_SUMMARIZER_PROMPT, followup_content)
                summarized_answer = final_response
        
        # Update the subtask with the summarized answer
        subtasks[idx] = SubTask(question=current_task["question"], answer=summarized_answer, status="completed")
        
        return {**state, "subtasks": subtasks}
    
    def _check_more_tasks(self, state: AgentState) -> AgentState:
        """Move to next task."""
        return {**state, "current_task_idx": state["current_task_idx"] + 1, "rag_state": None}
    
    def _route_tasks(self, state: AgentState) -> str:
        """Check if more tasks need execution."""
        return "task_executor" if state["current_task_idx"] < len(state["subtasks"]) else "answer_summarizer"
    
    # ==================== NODE 4: RAG SUBGRAPH ====================
    def _rag_rephraser(self, state: AgentState) -> AgentState:
        """Rephrase query for better RAG retrieval."""
        rag_state = state["rag_state"]
        query = rag_state["original_query"]
        prev_queries = rag_state["rephrased_queries"]
        
        prompt = f"Original query: {query}"
        if prev_queries:
            prompt += f"\n\nPrevious attempts (not relevant enough):\n" + "\n".join(prev_queries)
        
        rephrased = self._ask_llm(RAG_REPHRASER_PROMPT, prompt).strip()
        
        return {
            **state,
            "rag_state": create_rag_state(
                query,
                retry_count=rag_state["retry_count"],
                rephrased_queries=prev_queries + [rephrased]
            )
        }
    
    def _rag_retrieval_and_relevance_checker(self, state: AgentState) -> AgentState:
        """Retrieve documents, check relevance, and store answer if done."""
        rag_state = state["rag_state"]
        query = rag_state["original_query"]
        search_query = rag_state["rephrased_queries"][-1] if rag_state["rephrased_queries"] else query
        
        # Retrieve chunks via tool calling
        chunks = self._retrieve_chunks(search_query)
        new_retry_count = rag_state["retry_count"] + 1
        
        if not chunks:
            # No chunks found - check if we should retry or give up
            if new_retry_count >= self.MAX_RAG_RETRIES:
                # Max retries reached, store "no info found" as answer
                subtasks = list(state["subtasks"])
                idx = state["current_task_idx"]
                subtasks[idx] = SubTask(question=query, answer="No relevant information found in the legal documents.", status="completed")
                return {
                    **state,
                    "subtasks": subtasks,
                    "rag_state": {**rag_state, "retrieved_chunks": [], "is_relevant": True, "retry_count": new_retry_count}
                }
            return {
                **state,
                "rag_state": {**rag_state, "retrieved_chunks": [], "is_relevant": False, "retry_count": new_retry_count}
            }
        
        # Check relevance
        chunks_text = "\n---\n".join(chunks[:5])
        response = self._ask_llm(
            RAG_RELEVANCE_CHECKER_PROMPT,
            f"Query: {query}\n\nChunks:\n{chunks_text}"
        )
        
        is_relevant = check_relevance_keyword(response)
        
        # If relevant or max retries reached, store the answer
        if is_relevant or new_retry_count >= self.MAX_RAG_RETRIES:
            subtasks = list(state["subtasks"])
            idx = state["current_task_idx"]
            answer = "\n".join(chunks[:5]) if chunks else "No relevant information found in the legal documents."
            subtasks[idx] = SubTask(question=query, answer=answer, status="completed")
            return {
                **state,
                "subtasks": subtasks,
                "rag_state": {**rag_state, "retrieved_chunks": chunks, "relevant_chunks": chunks[:5], "retry_count": new_retry_count, "is_relevant": True}
            }
        
        # Not relevant yet, will retry
        return {
            **state,
            "rag_state": {
                **rag_state,
                "retrieved_chunks": chunks,
                "relevant_chunks": [],
                "retry_count": new_retry_count,
                "is_relevant": False
            }
        }
    
    def _retrieve_chunks(self, search_query: str) -> List[str]:
        """Retrieve document chunks using tool calling."""
        llm_with_tools = self.llm.bind_tools([self.search_tool])
        
        response = llm_with_tools.invoke([
            SystemMessage(content="You are a legal document retrieval assistant. Use the search_legal_documents tool to find relevant information for the query."),
            HumanMessage(content=f"Search for: {search_query}")
        ])
        
        chunks = []
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "search_legal_documents":
                    result = self.search_tool.invoke(tool_call["args"])
                    if result and result != "No relevant documents found.":
                        chunks = result.split("\n")
        return chunks
    
    def _route_rag_relevance(self, state: AgentState) -> str:
        """Route based on RAG relevance."""
        rag_state = state["rag_state"]
        if rag_state["is_relevant"]:
            return "subtask_summarizer"
        return "rag_rephraser"
    
    # ==================== NODE 5: ANSWER SUMMARIZER ====================
    def _answer_summarizer(self, state: AgentState) -> AgentState:
        """Summarize all subtask answers into final answer, with optional date calculation."""
        answers = [t["answer"] for t in state["subtasks"] if t["answer"]]
        chat_history = state.get("chat_history", [])
        
        if not answers:
            return {**state, "final_answer": "I couldn't find an answer to your question."}
        
        # Build user content with chat history for context-aware summarization
        user_content = f"Original Question: {state['original_question']}\n\nSubtask Answers:\n{chr(10).join(answers)}"
        if chat_history:
            history_context = self._format_chat_history(chat_history)
            user_content = f"Previous conversation (for context - helps understand follow-up questions):\n{history_context}\n\n{user_content}"
        
        # Use tool calling to allow date calculations during summarization
        llm_with_tools = self.llm.bind_tools([self.date_tool])
        response = llm_with_tools.invoke([
            SystemMessage(content=ANSWER_SUMMARIZER_PROMPT),
            HumanMessage(content=user_content)
        ])
        
        # Process any tool calls (date calculations)
        final_answer = response.content or ""
        if response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                if tool_call["name"] == "calculate_date_difference":
                    result = self.date_tool.invoke(tool_call["args"])
                    tool_results.append(result)
            
            # If there were tool calls, ask LLM to incorporate results
            if tool_results:
                followup_content = f"Original response: {final_answer}\n\nDate calculation results:\n{chr(10).join(tool_results)}\n\nPlease incorporate these date calculations into your final response."
                final_response = self._ask_llm(ANSWER_SUMMARIZER_PROMPT, followup_content)
                final_answer = final_response
        
        return {**state, "final_answer": final_answer}
    
    # ==================== NODE 6: COMPLETENESS CHECK ====================
    def _answer_completeness_check(self, state: AgentState) -> AgentState:
        """Check if all questions have answers."""
        response = self._ask_llm(
            ANSWER_COMPLETENESS_PROMPT,
            f"Original: {state['original_question']}\nAnswer: {state['final_answer']}"
        )
        
        if not check_relevance_keyword(response.replace("COMPLETE", "RELEVANT").replace("INCOMPLETE", "NOT_RELEVANT")):
            subtasks = list(state["subtasks"])
            for i, task in enumerate(subtasks):
                if not task["answer"] or "No relevant information" in (task["answer"] or ""):
                    subtasks[i] = SubTask(question=task["question"], answer=task["answer"], status="failed")
            return {**state, "subtasks": subtasks}
        
        return state
    
    def _route_completeness(self, state: AgentState) -> str:
        """Route based on completeness."""
        failed = [t for t in state["subtasks"] if t["status"] == "failed"]
        return "replan" if failed and state["replanning_count"] < self.MAX_REPLANS else "end"
    
    def _replan(self, state: AgentState) -> AgentState:
        """Increment replan counter and go back to planner."""
        return {**state, "replanning_count": state["replanning_count"] + 1}
    
    # ==================== BUILD GRAPH ====================
    def _build_graph(self) -> StateGraph:
        """Build the complete workflow graph."""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        nodes = {
            "relevancy_checker": self._relevancy_checker,
            "generic_response": self._generic_response,
            "task_planner": self._task_planner,
            "task_executor": self._task_executor,
            "subtask_summarizer": self._subtask_summarizer,
            "check_more_tasks": self._check_more_tasks,
            "rag_rephraser": self._rag_rephraser,
            "rag_retrieval_and_relevance_checker": self._rag_retrieval_and_relevance_checker,
            "answer_summarizer": self._answer_summarizer,
            "completeness_check": self._answer_completeness_check,
            "replan": self._replan,
        }
        for name, func in nodes.items():
            workflow.add_node(name, func)
        
        # Entry point
        workflow.set_entry_point("relevancy_checker")
        
        # Define edges
        workflow.add_conditional_edges("relevancy_checker", self._route_relevancy, 
                                       {"task_planner": "task_planner", "generic_response": "generic_response"})
        workflow.add_edge("generic_response", END)
        workflow.add_edge("task_planner", "task_executor")
        workflow.add_conditional_edges("task_executor", self._route_after_executor,
                                       {"task_planner": "task_planner", "rag_rephraser": "rag_rephraser", 
                                        "subtask_summarizer": "subtask_summarizer"})
        workflow.add_edge("subtask_summarizer", "check_more_tasks")
        workflow.add_conditional_edges("check_more_tasks", self._route_tasks,
                                       {"task_executor": "task_executor", "answer_summarizer": "answer_summarizer"})
        workflow.add_edge("rag_rephraser", "rag_retrieval_and_relevance_checker")
        workflow.add_conditional_edges("rag_retrieval_and_relevance_checker", self._route_rag_relevance,
                                       {"subtask_summarizer": "subtask_summarizer", "rag_rephraser": "rag_rephraser"})
        workflow.add_edge("answer_summarizer", "completeness_check")
        workflow.add_conditional_edges("completeness_check", self._route_completeness, 
                                       {"replan": "replan", "end": END})
        workflow.add_edge("replan", "task_planner")
        
        result = workflow.compile()
        self._save_graph_visualization(result)
        return result
    
    def _save_graph_visualization(self, graph):
        """Save graph visualization to file."""
        try:
            with open("./images/graph.png", "wb") as f:
                f.write(graph.get_graph().draw_mermaid_png())
        except Exception:
            pass
    
    # ==================== The chat flow ====================
    @observe(name="chat")
    def invoke(self, user_message: str, chat_history: List[BaseMessage] = None) -> str:
        """Invoke the workflow and return final answer.
        
        Args:
            user_message: The current user message/question
            chat_history: List of previous messages (HumanMessage/AIMessage) for context
        """
        config = RunnableConfig(callbacks=[self.langfuse_handler])
        initial_state = create_initial_state(user_message, chat_history)
        final_state = self.graph.invoke(initial_state, config=config)
        return final_state.get("final_answer", "I couldn't generate a response.")
    
    @observe(name="chat")
    def stream(self, user_message: str, chat_history: List[BaseMessage] = None):
        """Stream the workflow, yielding node events and final answer.
        
        Args:
            user_message: The current user message/question
            chat_history: List of previous messages (HumanMessage/AIMessage) for context
        """
        config = RunnableConfig(callbacks=[self.langfuse_handler])
        initial_state = create_initial_state(user_message, chat_history)
        final_answer = ""
        
        for event in self.graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                yield {"type": "node", "node": node_name, "display_name": NODE_DISPLAY_NAMES.get(node_name, node_name)}
                
                if isinstance(node_output, dict) and node_output.get("final_answer"):
                    final_answer = node_output["final_answer"]
        
        yield {"type": "answer", "content": final_answer or "I couldn't generate a response."}
