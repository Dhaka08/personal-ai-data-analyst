import io
import os
import contextlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Personal AI Data Analyst", layout="wide")
st.title("📊 Personal AI Data Analyst (Chat with CSV)")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to start.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

system_prompt = """
You are a Data Analyst.
You will be given a pandas dataframe named df.

Write ONLY valid Python pandas/matplotlib code to answer the user question.
Return ONLY raw Python code. Do NOT wrap the code in ``` or markdown.
Rules:
- Do NOT use print() unless necessary.
- Always store the final output in a variable named result.
- Do NOT write explanations.
- Do NOT import anything.
- Use print() to show text outputs.
- If a table is needed, store it in a variable named result.
- If a chart is needed, use matplotlib with plt.figure() and plt.show().
"""

def clean_code(code: str) -> str:
    return code.replace("```python", "").replace("```", "").strip()

question = st.text_input("💬 Ask a question about your dataset:")

if question:
    user_prompt = f"""
Columns: {list(df.columns)}
Question: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    code = clean_code(response.choices[0].message.content)

    st.subheader("🤖 AI Generated Code")
    st.code(code, language="python")

    st.subheader("✅ Output")

try:
    plt.close("all")
    exec_globals = {"df": df, "pd": pd, "plt": plt}

    # ✅ Capture print output
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(code, exec_globals)

    printed_output = buffer.getvalue().strip()

    # ✅ Show printed output
    if printed_output:
        st.text(printed_output)

    # ✅ Show result variable if exists
    if "result" in exec_globals:
        res = exec_globals["result"]
        if isinstance(res, (pd.DataFrame, pd.Series)):
            st.dataframe(res, use_container_width=True)
        else:
            st.write(res)

    # ✅ Show plot if generated
    st.pyplot(plt.gcf())

except Exception as e:
    st.error(f"❌ Error while running code: {e}")
