from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

from nodes import run_agent_reasoning_engine, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning_engine)
flow.add_node(ACT, tool_node)
flow.set_entry_point(AGENT_REASON)


def should_continue(state: dict) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


flow.add_edge(START, AGENT_REASON)
flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {
        END: END,
        ACT: ACT,
    },
)
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="what is the weather in Nagpur, Maharashtra, India? List it and then Triple it "
                )
            ]
        }
    )
    print(res["messages"][LAST].content)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
