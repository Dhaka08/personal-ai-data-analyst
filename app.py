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

# ✅ Also allow user to enter key from sidebar (optional)
st.sidebar.header("🔑 API Key")
user_key = st.sidebar.text_input("Enter OpenAI API Key (optional)", type="password")

final_key = user_key.strip() if user_key else API_KEY

if not final_key:
    st.warning("⚠️ Please set OPENAI_API_KEY in .env OR enter API key in sidebar.")
    st.stop()

client = OpenAI(api_key=final_key)

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
    """Remove markdown code fences if any."""
    return code.replace("```python", "").replace("```", "").strip()

# -------------------- UPLOAD CSV --------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to start.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

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
    """
    Executes AI-generated code safely in a limited global scope.
    Captures printed output and returns: (printed_text, result_obj, fig_if_any)
    """
    plt.close("all")

    exec_globals = {"df": df, "pd": pd, "plt": plt}

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(code, exec_globals)

    printed_output = buffer.getvalue().strip()
    result_obj = exec_globals.get("result", None)

    fig = plt.gcf()
    has_plot = bool(fig.axes)  # ✅ plot exists only if axes present
    return printed_output, result_obj, fig if has_plot else None

# -------------------- AUTO REPORT (Optional Button) --------------------
st.subheader("📌 Auto Insights Report")

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

# -------------------- ASK QUESTIONS --------------------
st.subheader("💬 Ask Questions About Your Data")
question = st.text_input("Ask a question about your dataset:")

# ✅ IMPORTANT: only run when question is not empty
if question and question.strip():
    with st.spinner("🤖 Generating code..."):
        code = generate_code(question.strip())

    st.subheader("🧠 AI Generated Code")
    st.code(code, language="python")

    st.subheader("✅ Output")

    # Attempt 1
    try:
        printed_output, result_obj, fig = execute_code(code)

        # Show printed output (if any)
        if printed_output:
            st.text(printed_output)

        # Show result variable (if any)
        if result_obj is not None:
            if isinstance(result_obj, (pd.DataFrame, pd.Series)):
                st.dataframe(result_obj, use_container_width=True)
            else:
                st.write(result_obj)

        # Show plot only if exists
        if fig is not None:
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error while running code: {e}")
        st.info("🔁 Trying to auto-fix...")

        # Attempt 2 (auto-fix)
        try:
            fixed_code = fix_code(question.strip(), code, str(e))
            st.subheader("🛠️ Fixed Code (Auto)")
            st.code(fixed_code, language="python")

            printed_output, result_obj, fig = execute_code(fixed_code)

            if printed_output:
                st.text(printed_output)

            if result_obj is not None:
                if isinstance(result_obj, (pd.DataFrame, pd.Series)):
                    st.dataframe(result_obj, use_container_width=True)
                else:
                    st.write(result_obj)

            if fig is not None:
                st.pyplot(fig)

        except Exception as e2:
            st.error(f"❌ Still failing after auto-fix: {e2}")
