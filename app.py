import os
import io
import contextlib

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Personal AI Data Analyst", layout="wide")
st.title("📊 Personal AI Data Analyst (Chat with CSV)")

# -------------------- LOAD ENV + CLIENT --------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

st.sidebar.header("🔑 API Key")

# ✅ If .env key exists, do NOT force user to enter anything
if API_KEY:
    st.sidebar.success("✅ API Key loaded from .env")
    final_key = API_KEY
else:
    user_key = st.sidebar.text_input("Enter OpenAI API Key", type="password").strip()

    if not user_key:
        st.warning("⚠️ Please enter your OpenAI API key to continue.")
        st.stop()

    # ✅ simple validation
    if not user_key.startswith("sk-"):
        st.error("❌ Invalid API key format. Key usually starts with 'sk-'.")
        st.stop()

    final_key = user_key

client = OpenAI(api_key=final_key)

# -------------------- SESSION STATE --------------------
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dicts: {question, code, output_text, table_obj, has_plot}

# -------------------- PROMPT --------------------
system_prompt = """
You are a Data Analyst.
You will be given a pandas dataframe named df.

Write ONLY valid Python pandas/matplotlib code to answer the user question.
Return ONLY raw Python code. Do NOT wrap the code in ``` or markdown.

Rules:
- Do NOT write explanations.
- Do NOT import anything.
- Avoid print() unless necessary.
- Always store the final output in a variable named result when possible.
- If a chart is needed, use matplotlib with plt.figure() and plt.show().
"""

def clean_code(code: str) -> str:
    return code.replace("```python", "").replace("```", "").strip()

# -------------------- UPLOAD FILE --------------------
uploaded_file = st.file_uploader(
    "📂 Upload your dataset",
    type=["csv", "xlsx", "xls", "json", "tsv", "txt"]
)

if uploaded_file is None:
    st.info("👆 Please upload a file to continue.")
    st.stop()

file_name = uploaded_file.name.lower()

try:
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif file_name.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep="\t")

    elif file_name.endswith(".txt"):
        # Try comma first, fallback to tab
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep="\t")

    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)

    elif file_name.endswith(".json"):
        df = pd.read_json(uploaded_file)

    else:
        st.error("❌ Unsupported file format.")
        st.stop()

except Exception as e:
    st.error(f"❌ Failed to read file: {e}")
    st.stop()

# -------------------- DATA PREVIEW --------------------
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# -------------------- DYNAMIC QUESTIONS (OPTION 2) --------------------
def get_dynamic_questions(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    questions = [
        "What are the names of all columns?",
        "Show the first 5 rows of the dataset",
        "How many rows and columns are there?",
        "Which columns have missing values? (store in result)",
    ]

    # Numeric-based questions
    if numeric_cols:
        questions += [
            "Show summary statistics for numeric columns (store in result)",
            f"Plot histogram of {numeric_cols[0]}",
        ]

        if len(numeric_cols) >= 2:
            questions += [
                "Plot correlation heatmap for numeric columns"
            ]

    # Categorical-based questions
    if cat_cols:
        questions += [
            f"Show top 10 most frequent values in {cat_cols[0]} (store in result)",
            f"Plot bar chart of top 10 values in {cat_cols[0]}",
        ]

    # Mixed analysis if both available
    if numeric_cols and cat_cols:
        questions += [
            f"Show average of {numeric_cols[0]} grouped by {cat_cols[0]} (store in result)"
        ]

    return questions

# -------------------- FUNCTIONS --------------------
def generate_code(question: str) -> str:
    user_prompt = f"""
Dataframe columns: {list(df.columns)}
User question: {question}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return clean_code(response.choices[0].message.content)

def fix_code(question: str, code: str, error: str) -> str:
    fix_prompt = f"""
The code below caused an error.

Question: {question}

Code:
{code}

Error:
{error}

Fix the code. Return ONLY corrected raw Python code (no markdown).
Remember: store the final output in a variable named result when possible.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": fix_prompt},
        ],
        temperature=0,
    )
    return clean_code(response.choices[0].message.content)

def execute_code(code: str):
    plt.close("all")
    exec_globals = {"df": df, "pd": pd, "plt": plt}

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(code, exec_globals)

    printed_output = buffer.getvalue().strip()
    result_obj = exec_globals.get("result", None)

    fig = plt.gcf()
    has_plot = bool(fig.axes)
    return printed_output, result_obj, fig if has_plot else None

# -------------------- AUTO REPORT --------------------
st.subheader("📌 Auto Insights Report")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Generate Report"):
        st.write("✅ Dataset Overview")
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        st.write("Columns:", list(df.columns))

        st.write("✅ Missing Values (Top)")
        missing = df.isnull().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if len(missing) == 0:
            st.success("No missing values ✅")
        else:
            st.dataframe(missing.head(10), use_container_width=True)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            st.write("✅ Numeric Summary")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

with col2:
    if st.button("Clear Chat"):
        st.session_state.chat = []
        st.success("✅ Chat cleared!")

# -------------------- CHAT UI --------------------
st.subheader("💬 Ask Questions About Your Data")

with st.expander("✅ Example Questions (Auto-generated for your dataset)"):
    qs = get_dynamic_questions(df)
    for q in qs:
        st.markdown(f"- {q}")

st.caption("💡 Tip: Ask AI to store final table output in `result` for best display.")

question = st.text_input("Ask a question (chat history will be saved):")

if question and question.strip():
    q = question.strip()

    with st.spinner("🤖 Generating code..."):
        code = generate_code(q)

    output_text = ""
    table_obj = None
    plot_fig = None

    # Attempt 1
    try:
        printed_output, result_obj, fig = execute_code(code)
        output_text = printed_output
        table_obj = result_obj
        plot_fig = fig

    except Exception as e:
        # Attempt 2 auto-fix
        try:
            fixed_code = fix_code(q, code, str(e))
            code = fixed_code

            printed_output, result_obj, fig = execute_code(code)
            output_text = printed_output
            table_obj = result_obj
            plot_fig = fig

        except Exception as e2:
            output_text = f"❌ Still failing after auto-fix: {e2}"

    # ✅ Save in chat history
    st.session_state.chat.append(
        {
            "question": q,
            "code": code,
            "output_text": output_text,
            "table_obj": table_obj,
            "has_plot": plot_fig is not None,
        }
    )

# -------------------- DISPLAY CHAT HISTORY --------------------
st.markdown("## 🧾 Chat History")

if len(st.session_state.chat) == 0:
    st.info("No questions asked yet. Start by typing a question above.")
else:
    for i, item in enumerate(reversed(st.session_state.chat), start=1):
        st.markdown(f"### 🧑 Question {len(st.session_state.chat)-i+1}: {item['question']}")
        st.markdown("**🤖 Generated Code:**")
        st.code(item["code"], language="python")

        st.markdown("**✅ Output:**")
        if item["output_text"]:
            st.text(item["output_text"])

        if item["table_obj"] is not None:
            if isinstance(item["table_obj"], (pd.DataFrame, pd.Series)):
                st.dataframe(item["table_obj"], use_container_width=True)
            else:
                st.write(item["table_obj"])

        # Note: plot is not stored to avoid memory heavy behavior
        if item["has_plot"]:
            st.info("📌 Plot generated (not stored in history to save memory). Ask again to view plot.")

        st.markdown("---")
