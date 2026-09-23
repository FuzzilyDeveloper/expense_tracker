from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


st.set_page_config(
    page_title="Luma Finance",
    page_icon="◒",
    layout="wide",
    initial_sidebar_state="expanded",
)


CATEGORIES = ["Food", "Transport", "Home", "Health", "Shopping", "Fun", "Other"]
SEED_EXPENSES = [
    {"id": "1", "date": "2026-09-21", "merchant": "Morrow Market", "category": "Food", "amount": 42.80, "note": "Weekly groceries"},
    {"id": "2", "date": "2026-09-20", "merchant": "Metro Transit", "category": "Transport", "amount": 18.00, "note": "Monthly pass"},
    {"id": "3", "date": "2026-09-18", "merchant": "Northstar Pharmacy", "category": "Health", "amount": 27.45, "note": "Vitamins"},
    {"id": "4", "date": "2026-09-16", "merchant": "Tide & Timber", "category": "Home", "amount": 64.00, "note": "Kitchen towels"},
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink:#18231f; --muted:#6c7973; --mint:#b9e4c9; --lime:#d8f06f; --cream:#f7f5ee; --line:#dce4dc; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color:var(--ink); }
        .stApp { background:var(--cream); }
        [data-testid="stSidebar"] { background:#18231f; border-right:0; }
        [data-testid="stSidebar"] * { color:#eef4ed; }
        [data-testid="stSidebar"] .stCaption { color:#a8b8ae; }
        h1, h2, h3 { font-family:'Space Grotesk', sans-serif; letter-spacing:0; }
        .hero { background:linear-gradient(120deg,#d9f0df 0%,#c4e9cf 55%,#b3dfc3 100%); padding:2.4rem 2.6rem 2.2rem; border-radius:22px; margin-bottom:1.3rem; position:relative; overflow:hidden; }
        .hero:after { content:'◒'; position:absolute; right:3rem; top:-2.1rem; font-size:12rem; line-height:1; color:rgba(24,35,31,.08); transform:rotate(-12deg); }
        .eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.7rem; font-weight:700; color:#48715a; margin-bottom:.8rem; }
        .hero h1 { font-size:clamp(2rem,4vw,3.35rem); margin:0; color:#18231f; position:relative; z-index:1; }
        .hero p { color:#486254; font-size:1.02rem; margin:.7rem 0 0; max-width:33rem; position:relative; z-index:1; }
        .metric { background:#fff; border:1px solid var(--line); padding:1.15rem 1.25rem; border-radius:16px; min-height:105px; }
        .metric-label { color:var(--muted); font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; }
        .metric-value { font-family:'Space Grotesk'; font-size:1.8rem; font-weight:700; margin-top:.35rem; }
        .section-label { color:#5b6d62; font-size:.75rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin:1.5rem 0 .65rem; }
        .expense-row { display:flex; align-items:center; gap:1rem; background:#fff; border-bottom:1px solid #edf0eb; padding:.9rem 1rem; }
        .expense-row:first-child { border-radius:14px 14px 0 0; } .expense-row:last-child { border-radius:0 0 14px 14px; border-bottom:0; }
        .expense-icon { width:38px; height:38px; border-radius:12px; background:#e5f3e8; display:grid; place-items:center; font-size:1.15rem; }
        .expense-name { font-weight:700; font-size:.92rem; } .expense-meta { color:var(--muted); font-size:.78rem; margin-top:.15rem; }
        .expense-amount { margin-left:auto; font-family:'Space Grotesk'; font-weight:700; }
        .stButton > button { border-radius:10px; border:1px solid #cbd8cc; font-weight:600; }
        .stButton > button[kind="primary"] { background:#18231f; color:white; border:0; }
        div[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def money(value: float) -> str:
    return f"${value:,.2f}"


def get_expenses() -> list[dict[str, Any]]:
    if "expenses" not in st.session_state:
        st.session_state.expenses = [item.copy() for item in SEED_EXPENSES]
    return st.session_state.expenses


def category_icon(category: str) -> str:
    return {"Food": "◉", "Transport": "↗", "Home": "⌂", "Health": "+", "Shopping": "◇", "Fun": "✦"}.get(category, "•")


def build_agent(expenses: list[dict[str, Any]], model_name: str, api_key: str) -> AgentExecutor:
    serialized = json.dumps(expenses, default=str)

    @tool
    def spending_summary() -> str:
        """Return total spending grouped by category from the current expense list."""
        totals: dict[str, float] = {}
        for expense in expenses:
            category = str(expense["category"])
            totals[category] = totals.get(category, 0) + float(expense["amount"])
        return json.dumps({key: round(value, 2) for key, value in sorted(totals.items(), key=lambda item: -item[1])})

    @tool
    def recent_transactions(limit: int = 5) -> str:
        """Return the most recent transactions, up to the requested limit."""
        ordered = sorted(expenses, key=lambda item: item["date"], reverse=True)
        return json.dumps(ordered[: max(1, min(limit, 10))])

    @tool
    def all_expenses() -> str:
        """Return all current expenses as JSON for detailed questions."""
        return serialized

    tools = [spending_summary, recent_transactions, all_expenses]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are Luma, a concise and thoughtful personal finance assistant. Use the provided tools for every question about the user's spending. Never invent transactions. Give practical observations and format currency in USD.\nCurrent date: {today}"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm = ChatOpenAI(model=model_name, temperature=0.2, api_key=api_key)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


def render_expenses(expenses: list[dict[str, Any]]) -> None:
    if not expenses:
        st.info("No expenses yet. Add your first one in the sidebar.")
        return
    rows = sorted(expenses, key=lambda item: item["date"], reverse=True)
    html = "".join(
        f'<div class="expense-row"><div class="expense-icon">{category_icon(item["category"])}</div>'
        f'<div><div class="expense-name">{item["merchant"]}</div><div class="expense-meta">{item["category"]} · {item["date"]}</div></div>'
        f'<div class="expense-amount">{money(float(item["amount"]))}</div></div>'
        for item in rows
    )
    st.markdown(html, unsafe_allow_html=True)


inject_styles()
expenses = get_expenses()

with st.sidebar:
    st.markdown("## ◒ Luma Finance")
    st.caption("A calm place for your everyday money.")
    st.divider()
    st.markdown("### Add an expense")
    with st.form("expense_form", clear_on_submit=True):
        merchant = st.text_input("Merchant", placeholder="e.g. Corner Cafe")
        amount = st.number_input("Amount", min_value=0.01, value=12.50, step=0.50)
        category = st.selectbox("Category", CATEGORIES)
        expense_date = st.date_input("Date", value=date.today())
        note = st.text_input("Note", placeholder="Optional")
        submitted = st.form_submit_button("Add expense", type="primary", use_container_width=True)
        if submitted:
            if not merchant.strip():
                st.error("Add a merchant name first.")
            else:
                expenses.append({"id": datetime.now().isoformat(), "date": expense_date.isoformat(), "merchant": merchant.strip(), "category": category, "amount": float(amount), "note": note.strip()})
                st.success("Expense added.")
                st.rerun()
    st.divider()
    st.markdown("### Luma assistant")
    api_key = st.text_input("OpenAI API key", type="password", help="Used only for this Streamlit session.")
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], label_visibility="collapsed")
    if not api_key:
        st.caption("Add a key to ask questions about your spending with LangChain.")

today_label = date.today().strftime("%A, %B %d, %Y").replace(" 0", " ")
st.markdown(f'<div class="hero"><div class="eyebrow">{today_label}</div><h1>Money, made visible.</h1><p>See the shape of your spending, then make your next move with intention.</p></div>', unsafe_allow_html=True)

total = sum(float(item["amount"]) for item in expenses)
largest = max((float(item["amount"]) for item in expenses), default=0)
categories_total = len({item["category"] for item in expenses})
metric_cols = st.columns(3)
for column, label, value in zip(metric_cols, ["Tracked this month", "Largest expense", "Categories used"], [money(total), money(largest), str(categories_total)]):
    with column:
        st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

left, right = st.columns([1.3, 1], gap="large")
with left:
    st.markdown('<div class="section-label">Recent activity</div>', unsafe_allow_html=True)
    render_expenses(expenses)
with right:
    st.markdown('<div class="section-label">Ask Luma</div>', unsafe_allow_html=True)
    question = st.chat_input("Where did I spend the most?")
    if question:
        if not api_key:
            st.warning("Add an OpenAI API key in the sidebar to start the LangChain assistant.")
        else:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Reviewing your expenses..."):
                    try:
                        result = build_agent(expenses, model_name, api_key).invoke({"input": question, "today": date.today().isoformat()})
                        st.write(result["output"])
                    except Exception as error:
                        st.error(f"The assistant could not respond: {error}")
    else:
        st.info("Try asking: “How much did I spend on Food?”")

    st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
    csv_lines = ["date,merchant,category,amount,note"] + [
        f'{item["date"]},"{item["merchant"].replace(chr(34), chr(34) * 2)}",{item["category"]},{item["amount"]},"{item["note"].replace(chr(34), chr(34) * 2)}"'
        for item in expenses
    ]
    st.download_button("Download CSV", "\n".join(csv_lines), "luma-expenses.csv", "text/csv", use_container_width=True)