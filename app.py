import os
import io
import json
import time
import contextlib

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Personal AI Data Analyst | Chat with Data", layout="wide")
st.title("📊 Personal AI Data Analyst (Chat with Data)")

# -------------------- LOAD ENV + CLIENT --------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

st.sidebar.header("🔑 API Key")

client = None

if API_KEY:
    st.sidebar.success("✅ API Key loaded from secrets")
    client = OpenAI(api_key=API_KEY)
else:
    user_key = st.sidebar.text_input("Enter OpenAI API Key", type="password").strip()
    if user_key:
        if not user_key.startswith("sk-"):
            st.error("❌ Invalid API key format.")
            st.stop()
        client = OpenAI(api_key=user_key)
    else:
        st.sidebar.info("ℹ️ AI features disabled (Offline Mode)")

# -------------------- SESSION STATE --------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------- PROMPT --------------------
system_prompt = """
You are a Data Analyst.
You will be given a pandas dataframe named df.

Write ONLY valid Python pandas/matplotlib code.
No imports, file access, network calls, or OS commands.
Store final table output in variable named `result` when possible.
"""

def clean_code(code: str) -> str:
    return code.replace("```python", "").replace("```", "").strip()

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "📂 Upload your dataset",
    type=["csv", "xlsx", "xls", "json", "tsv", "txt"]
)

if uploaded_file is None:
    st.info("👆 Upload a file to continue.")
    st.stop()

file_name = uploaded_file.name.lower()

try:
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep="\t")
    elif file_name.endswith(".txt"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif file_name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# -------------------- DATA PREVIEW --------------------
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

with st.expander("📌 Dataset Info"):
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.dataframe(df.dtypes.astype(str), use_container_width=True)

# -------------------- SAFETY --------------------
def is_code_safe(code: str):
    blocked = ["import", "os.", "sys.", "open(", "exec(", "eval(", "subprocess"]
    for b in blocked:
        if b in code.lower():
            return False, b
    return True, None

# -------------------- OPENAI CALL --------------------
def generate_code(question: str) -> str:
    if client is None:
        st.warning("🔒 AI mode is disabled. Please provide your own OpenAI API key.")
        st.stop()

    user_prompt = f"""
Columns: {list(df.columns)}
Dtypes: {df.dtypes.astype(str).to_dict()}
Question: {question}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        return clean_code(response.choices[0].message.content)

    except RateLimitError as e:
        # covers insufficient_quota + rate limit
        st.error(
            "⚠️ OpenAI quota exceeded.\n\n"
            "➡️ This app requires an API key with available credits.\n"
            "➡️ Please add your own key or use Offline Mode."
        )
        st.stop()

    except Exception as e:
        st.error(f"❌ OpenAI error: {e}")
        st.stop()

# -------------------- EXECUTION --------------------
def execute_code(code: str):
    safe, bad = is_code_safe(code)
    if not safe:
        raise ValueError(f"Blocked unsafe code: {bad}")

    plt.close("all")
    exec_globals = {"df": df, "pd": pd, "plt": plt}

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(code, exec_globals)

    output = buffer.getvalue().strip()
    result = exec_globals.get("result")
    fig = plt.gcf() if plt.get_fignums() else None
    return output, result, fig

def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def to_csv_bytes(obj):
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        return obj.to_csv(index=False).encode("utf-8")
    return None

# -------------------- CHAT UI --------------------
# -------------------- CHAT UI --------------------
st.subheader("💬 Ask Questions About Your Data")

st.sidebar.markdown("### ⚙️ Mode")
offline_mode = st.sidebar.toggle(
    "Offline Mode (No API)",
    value=(client is None),
    help="Turn OFF to enable AI chat using your OpenAI API key"
)

if offline_mode:
    st.sidebar.info("✅ Offline Mode enabled. AI chat is disabled.")

question = st.text_input(
    "Ask a question:",
    disabled=offline_mode,
    placeholder="Enable AI Mode to ask questions" if offline_mode else ""
)

# -------------------- AI CHAT LOGIC --------------------
if question and not offline_mode:
    with st.spinner("🤖 Generating answer..."):
        code = generate_code(question)

    try:
        out, table, fig = execute_code(code)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # ---------- TEXT OUTPUT ----------
    if out:
        st.text(out)

    # ---------- TABLE OUTPUT ----------
    if isinstance(table, (pd.DataFrame, pd.Series)):
        st.dataframe(table, use_container_width=True)

        csv = to_csv_bytes(table)
        if csv:
            st.download_button(
                label="⬇️ Download Result as CSV",
                data=csv,
                file_name="result.csv",
                mime="text/csv",
                key=f"csv_{len(st.session_state.chat)}"
            )

    # ---------- PLOT OUTPUT ----------
    plot_bytes = None
    if fig:
        st.pyplot(fig)

        plot_bytes = fig_to_png(fig)
        st.download_button(
            label="⬇️ Download Plot as PNG",
            data=plot_bytes,
            file_name="plot.png",
            mime="image/png",
            key=f"plot_{len(st.session_state.chat)}"
        )

    # ---------- SAVE CHAT ----------
    st.session_state.chat.append({
        "question": question,
        "code": code,
        "output": out,
        "table": table,
        "plot": plot_bytes,
    })

# -------------------- OFFLINE MODE MESSAGE --------------------
elif offline_mode:
    st.info(
        "🔒 AI Chat is disabled in Offline Mode.\n\n"
        "➡️ To use AI features, turn OFF Offline Mode and provide a valid OpenAI API key."
    )


# -------------------- CHAT HISTORY --------------------
st.markdown("## 🧾 Chat History")

for i, item in enumerate(reversed(st.session_state.chat), start=1):
    st.markdown(f"### Q{i}: {item['question']}")
    st.code(item["code"], language="python")

    if item["output"]:
        st.text(item["output"])

    if isinstance(item["table"], (pd.DataFrame, pd.Series)):
        st.dataframe(item["table"], use_container_width=True)

    if item["plot"]:
        st.image(item["plot"], use_container_width=True)

    st.markdown("---")
