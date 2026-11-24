"""Word-meaning lab: entry point.  Run from the repository root:  streamlit run app/word_lab.py

Thin UI only; all data handling and decisions live in src/app/logic.py (unit-tested).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # trusted project code (src/)

import streamlit as st  # noqa: E402

st.set_page_config(page_title='Word-meaning lab', page_icon=':material/translate:', layout='wide')

pages = [
    st.Page('views/meaning.py', title='What does a word mean?',
            icon=':material/translate:', default=True),
    st.Page('views/simulator.py', title='Crisis-line simulator',
            icon=':material/support_agent:', url_path='simulator'),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown('**Same word, different meaning?**')
    st.caption('Compares what citizens (CAPphrase survey, 5,174 people) and four local LLMs mean by '
               'probability words such as "likely", and what that does to a crisis-line bot.')
    st.caption(':material/warning: Research prototype. The escalation rules are a design proposal and are '
               'NOT clinically validated. Do not use for real crisis decisions.')
    st.caption('Data: committed aggregate results in results/words/ only.')

nav.run()
