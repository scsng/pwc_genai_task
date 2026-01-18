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


def create_initial_state(user_message: str) -> AgentState:
    """Create the initial workflow state."""
    return {
        "messages": [HumanMessage(content=user_message)],
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
    "check_more_tasks": "✅ Checking Progress",
    "rag_rephraser": "✏️ Rephrasing Query",
    "rag_retrieval_and_relevance_checker": "📚 Searching Documents",
    "rag_to_executor": "📄 Processing Results",
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
    def _ask_llm(self, system_prompt: str, user_content: str) -> str:
        """Send a message to LLM and return the response content."""
        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ])
        return response.content
    
    # ==================== NODE 1: RELEVANCY CHECKER ====================
    def _relevancy_checker(self, state: AgentState) -> AgentState:
        """Check if question is relevant to our capabilities."""
        response = self._ask_llm(
            RELEVANCY_CHECKER_PROMPT,
            f"Question: {state['original_question']}"
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
        
        # Determine what question to plan
        if task_to_split:
            question_to_plan = task_to_split
        else:
            question_to_plan = state["original_question"]
            failed = [t["question"] for t in existing_subtasks if t["status"] == "failed"]
            if failed:
                question_to_plan += "\n\nPreviously failed:\n" + "\n".join(failed)
        
        # Get subtasks from LLM
        prompt = TASK_PLANNER_PROMPT.format(max_task_count=self.max_task_count)
        response = self._ask_llm(prompt, f"Question: {question_to_plan}")
        
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
        """Execute current subtask."""
        subtasks = list(state["subtasks"])
        idx = state["current_task_idx"]
        
        if idx >= len(subtasks):
            return state
        
        question = subtasks[idx]["question"]
        response = self._ask_llm(TASK_EXECUTOR_PROMPT, f"Task: {question}")
        
        # Check for split request
        if "NEED_SPLIT" in response:
            return {**state, "needs_split": True, "task_to_split": question}
        
        # Try date calculation
        answer = self._try_date_calculation(response, question)
        
        # Check if RAG is needed
        if answer is None and "NEED_RAG" in response:
            return {
                **state,
                "rag_state": create_rag_state(question),
                "needs_split": False
            }
        
        # Use direct answer if no tool was triggered
        if answer is None:
            answer = f"Q: {question}\nA: {response}"
        
        subtasks[idx] = SubTask(question=question, answer=answer, status="completed")
        return {**state, "subtasks": subtasks, "needs_split": False}
    
    def _try_date_calculation(self, response: str, question: str) -> Optional[str]:
        """Attempt to extract and calculate dates from response."""
        if "CALCULATE_DATE:" not in response:
            return None
        try:
            date_part = response.split("CALCULATE_DATE:")[1].strip()
            dates = date_part.split(",")
            if len(dates) >= 2:
                result = self.date_tool.invoke({
                    "date1": dates[0].strip(),
                    "date2": dates[1].strip()
                })
                return f"Q: {question}\nA: {result}"
        except Exception:
            pass
        return None
    
    def _route_after_executor(self, state: AgentState) -> str:
        """Route after task execution."""
        if state.get("needs_split"):
            return "task_planner"
        rag_state = state.get("rag_state")
        if rag_state and not rag_state.get("is_relevant", False):
            return "rag_rephraser"
        return "check_more_tasks"
    
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
        """Retrieve documents and check relevance."""
        rag_state = state["rag_state"]
        query = rag_state["original_query"]
        search_query = rag_state["rephrased_queries"][-1] if rag_state["rephrased_queries"] else query
        
        # Retrieve chunks via tool calling
        chunks = self._retrieve_chunks(search_query)
        
        if not chunks:
            return {
                **state,
                "rag_state": {**rag_state, "retrieved_chunks": [], "is_relevant": False, 
                              "retry_count": rag_state["retry_count"] + 1}
            }
        
        # Check relevance
        chunks_text = "\n---\n".join(chunks[:5])
        response = self._ask_llm(
            RAG_RELEVANCE_CHECKER_PROMPT,
            f"Query: {query}\n\nChunks:\n{chunks_text}"
        )
        
        is_relevant = check_relevance_keyword(response)
        
        return {
            **state,
            "rag_state": {
                **rag_state,
                "retrieved_chunks": chunks,
                "relevant_chunks": chunks[:5] if is_relevant else [],
                "retry_count": rag_state["retry_count"] + 1,
                "is_relevant": is_relevant
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
        if rag_state["is_relevant"] or rag_state["retry_count"] >= self.MAX_RAG_RETRIES:
            return "rag_to_executor"
        return "rag_rephraser"
    
    def _rag_to_executor(self, state: AgentState) -> AgentState:
        """Convert RAG results to task answer."""
        rag_state = state["rag_state"]
        subtasks = list(state["subtasks"])
        idx = state["current_task_idx"]
        
        query = rag_state["original_query"]
        chunks = rag_state["relevant_chunks"] or rag_state["retrieved_chunks"][:3]
        search_query = rag_state["rephrased_queries"][-1] if rag_state["rephrased_queries"] else query
        
        answer = (f"Q: {query}\nSearch Query: {search_query}\nA: " + "\n".join(chunks)) if chunks \
            else f"Q: {query}\nA: No relevant information found."
        
        subtasks[idx] = SubTask(question=query, answer=answer, status="completed")
        return {**state, "subtasks": subtasks, "rag_state": {**rag_state, "is_relevant": True}}
    
    # ==================== NODE 5: ANSWER SUMMARIZER ====================
    def _answer_summarizer(self, state: AgentState) -> AgentState:
        """Summarize all subtask answers into final answer."""
        answers = [t["answer"] for t in state["subtasks"] if t["answer"]]
        
        if not answers:
            return {**state, "final_answer": "I couldn't find an answer to your question."}
        
        response = self._ask_llm(
            ANSWER_SUMMARIZER_PROMPT,
            f"Original Question: {state['original_question']}\n\nSubtask Answers:\n{chr(10).join(answers)}"
        )
        return {**state, "final_answer": response}
    
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
            "check_more_tasks": self._check_more_tasks,
            "rag_rephraser": self._rag_rephraser,
            "rag_retrieval_and_relevance_checker": self._rag_retrieval_and_relevance_checker,
            "rag_to_executor": self._rag_to_executor,
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
                                        "check_more_tasks": "check_more_tasks"})
        workflow.add_conditional_edges("check_more_tasks", self._route_tasks,
                                       {"task_executor": "task_executor", "answer_summarizer": "answer_summarizer"})
        workflow.add_edge("rag_rephraser", "rag_retrieval_and_relevance_checker")
        workflow.add_conditional_edges("rag_retrieval_and_relevance_checker", self._route_rag_relevance,
                                       {"rag_to_executor": "rag_to_executor", "rag_rephraser": "rag_rephraser"})
        workflow.add_edge("rag_to_executor", "check_more_tasks")
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
            with open("graph.png", "wb") as f:
                f.write(graph.get_graph().draw_mermaid_png())
        except Exception:
            pass
    
    # ==================== The chat flow ====================
    @observe(name="chat")
    def invoke(self, user_message: str, chat_history: list = None) -> str:
        """Invoke the workflow and return final answer."""
        config = RunnableConfig(callbacks=[self.langfuse_handler])
        final_state = self.graph.invoke(create_initial_state(user_message), config=config)
        return final_state.get("final_answer", "I couldn't generate a response.")
    
    @observe(name="chat")
    def stream(self, user_message: str, chat_history: list = None):
        """Stream the workflow, yielding node events and final answer."""
        config = RunnableConfig(callbacks=[self.langfuse_handler])
        final_answer = ""
        
        for event in self.graph.stream(create_initial_state(user_message), config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                yield {"type": "node", "node": node_name, "display_name": NODE_DISPLAY_NAMES.get(node_name, node_name)}
                
                if isinstance(node_output, dict) and node_output.get("final_answer"):
                    final_answer = node_output["final_answer"]
        
        yield {"type": "answer", "content": final_answer or "I couldn't generate a response."}
