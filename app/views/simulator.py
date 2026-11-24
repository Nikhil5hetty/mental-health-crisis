"""Page 2: Crisis-line simulator  (thin UI over src/app/logic.py; rule shared with experiments/words/run_rq3.py)"""
import altair as alt
import pandas as pd
import streamlit as st

from src.app import logic
from src.cww import APPROPRIATENESS_WORDS


@st.cache_resource(show_spinner=False)
def simulator(results: str):
    return logic.load_simulator(results)


st.title('Crisis-line simulator')
st.info(
    '**How to read this.** A chatbot estimates how likely a message signals a crisis (the *crisis degree*, '
    '0-100) and turns that number into a word: no / possible / likely / clear crisis. Citizens and LLMs '
    'use different meanings for those words, so the same number can become a different word, and the '
    'word plus the quality of the bot\'s reply decides the action (A0 continue ... A3 urgent). '
    'Move the slider and see where the decisions split.', icon=':material/help:')

try:
    sim = simulator(str(logic.results_dir()))
except logic.MissingResults as exc:
    st.error(str(exc))
    st.stop()

left, right = st.columns([3, 2])
m = left.slider('Crisis degree (share of crisis signals, %)', 0.0, 100.0, 33.0, 0.5, format='%.1f',
                help='In the study: the share of LLM labellers that marked the message as a crisis.')
reply = right.selectbox('Quality of the bot\'s reply', APPROPRIATENESS_WORDS,
                        index=APPROPRIATENESS_WORDS.index('Mostly Appropriate'),
                        format_func=lambda w: f'{w}: {logic.REPLY_HELP[w]}')

table = logic.simulate(sim, m, reply)
h = logic.headline(table)
if h['llm_flips']:
    st.error(f':material/swap_vert: **Decision flips.** With citizen meanings the bot does '
             f'**{logic.ACTION_LABELS[h["citizen_action"]]}**; with LLM meanings it does '
             f'**{logic.ACTION_LABELS[h["llm_action"]]}** ({h["direction"]}).')
else:
    st.success(f':material/check_circle: **Same decision** for citizens and LLMs: '
               f'{logic.ACTION_LABELS[h["citizen_action"]]}.')
if h['n_groups_differ']:
    st.warning(f'{h["n_groups_differ"]} of {h["n_groups"]} citizen groups would decide differently from citizens overall.')

view = pd.DataFrame({
    'Whose meanings': table.label,
    'Crisis word chosen': [f'{a} ("{w}")' for a, w in zip(table.level_label, table.word)],
    'Fit (0-1)': table.membership.round(2),
    'Action': table.action_short,
    'Decision vs citizens': ['FLIPS (' + d + ')' if f else 'same' for f, d in zip(table.flips, table.direction)],
    'Note': ['no fit (*)' if fb else '' for fb in table.fallback],
})
flipped = table.flips.to_numpy()
styled = view.style.apply(lambda row: ['background-color: #fde2da' if flipped[row.name] else '' for _ in row], axis=1).format({'Fit (0-1)': '{:.2f}'})
st.dataframe(styled, hide_index=True, width='stretch')
st.caption('Actions: ' + ' | '.join(logic.ACTION_LABELS) + '. Highlighted rows decide differently from citizens overall.')
if h['any_fallback']:
    st.caption('(*) No fit: all four crisis words have fit 0 at this number, so the word whose centre is '
               'nearest is used (the same rule as the study, RQ3).')

with st.expander('Where does each crisis word start and stop?'):
    curves = logic.level_curves(sim)
    levels = list(dict.fromkeys(curves.level))
    rule = alt.Chart(pd.DataFrame({'x': [m]})).mark_rule(color='#0b0b0b', strokeDash=[4, 3]).encode(x='x:Q')
    for name, part in curves.groupby('codebook', sort=False):
        lines = alt.Chart(part).mark_line(strokeWidth=2).encode(
            x=alt.X('x:Q', title='Crisis degree (%)', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('mu:Q', title='Fit (0-1)'),
            color=alt.Color('level:N', title='Crisis word', sort=levels,
                            scale=alt.Scale(domain=levels, range=['#1baf7a', '#eda100', '#eb6834', '#6250d6'])),
            tooltip=['level:N', alt.Tooltip('x:Q', title='%'), alt.Tooltip('mu:Q', format='.2f', title='fit')])
        st.markdown(f'**{name}**')
        st.altair_chart(alt.layer(lines, rule).properties(height=160), width='stretch')
    st.caption('Gaps where no line is above 0 are crisis degrees that no word covers in that codebook.')

if sim.rq3:
    r = sim.rq3
    st.caption(f'In the study (RQ3), {r["n_flips"]} of {r["n_items"]} real crisis-line messages flipped between '
               f'citizen and LLM meanings: {r["flip_rate"]:.1%} (95% CI {r["ci_lo"]:.1%} to {r["ci_hi"]:.1%}).')
else:
    st.caption('Study results for RQ3 (results/words/rq3) were not found. The simulator still works: it only '
               'needs the word codebooks and the rule table.')
