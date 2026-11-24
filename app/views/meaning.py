"""Page 1: What does a word mean?  (thin UI over src/app/logic.py)"""
import altair as alt
import streamlit as st

from src.app import charts, logic

BLUE, ORANGE, GREY = charts.BLUE, charts.ORANGE, charts.GREY


@st.cache_data(show_spinner=False)
def data(results: str):
    return logic.load_word_data(results)


st.title('What does a word mean?')
st.caption('Each word is a fuzzy set on 0-100 %: how well each number fits the word (membership 0 to 1).')

try:
    d = data(str(logic.results_dir()))
except logic.MissingResults as exc:
    st.error(str(exc))
    st.stop()

tab_one, tab_rank = st.tabs(['One word', 'All 19 words ranked'])

with tab_one:
    left, right = st.columns([2, 1])
    term = left.selectbox('Probability word', d.terms, index=d.terms.index('Likely') if 'Likely' in d.terms else 0,
                          help='The 19 words of the CAPphrase survey.')
    show_models = right.toggle('Show each LLM', value=True, help='One line per model (question P1).')
    show_groups = right.toggle('Show the 6 citizen groups', value=False,
                               help='Native / non-native English speakers x three age bands.')
    s = logic.term_summary(d, term)

    c1, c2, c3 = st.columns(3)
    j, sh = s['jaccard'], s['centre_shift']
    c1.metric('Overlap citizens vs LLMs (Jaccard)', f'{j["estimate"]:.2f}',
              help=f'0 = no overlap, 1 = identical. 95% CI {j["ci_lo"]:.2f} to {j["ci_hi"]:.2f}.')
    c1.caption(logic.overlap_words(j['estimate']).capitalize() + f' (95% CI {j["ci_lo"]:.2f}-{j["ci_hi"]:.2f}).')
    c2.metric('Centre shift (LLM - citizens)', f'{sh["estimate"]:+.1f} pp',
              help=f'Percentage points. 95% CI {sh["ci_lo"]:+.1f} to {sh["ci_hi"]:+.1f}.')
    c2.caption(logic.shift_words(sh['estimate']) + '.')
    n_used = int(s['models'].included.sum())
    c3.metric('Models giving one fixed number', f'{len(s["collapsed_models"])} of {n_used}',
              help='"Collapsed": the model gave the same number in at least 90% of 50 answers, so its set is a spike.')
    c3.caption(', '.join(s['collapsed_models']) if s['collapsed_models'] else 'None: every model varies its answer.')
    if s['excluded_models']:
        st.warning(f'Left out for this word (fewer than 30 valid answers): {", ".join(s["excluded_models"])}.')

    bands, lines = logic.term_curves(d, term, show_groups=show_groups, show_models=show_models)
    st.altair_chart(charts.word_chart(bands, lines), width='stretch')
    st.caption('Solid outlines show where the word applies: the outer line is the widest group or model, the inner '
               'line is where they all agree. Light shading between the lines shows disagreement.')

    with st.expander('How to read this', expanded=False):
        st.markdown(
            '- **Blue outlines**: citizens. The outer (upper) line is the widest of the 6 demographic groups; the '
            'inner (lower) line is where all 6 agree. Shading between them shows the groups disagree.\n'
            '- **Orange outlines**: the four LLMs together (question P1, 50 answers each), read the same way.\n'
            '- **Dashed lines**: one LLM each. A tall thin spike means the model always gives the same number.\n'
            '- **Overlap (Jaccard)** compares the two bands; **centre shift** compares where they sit.\n'
            '- Ranges in brackets are 95% bootstrap confidence intervals from the study.')
    with st.expander('Per-model details'):
        table = s['models'].rename(columns={'model': 'LLM', 'included': 'used', 'n_valid': 'valid answers',
                                            'n_asked': 'asked', 'collapsed': 'one fixed number',
                                            'jaccard': 'overlap with citizens', 'centre_shift': 'centre shift (pp)',
                                            'width_ratio': 'spread ratio (LLM / citizens)'})
        st.dataframe(table, hide_index=True, width='stretch')

with tab_rank:
    r = logic.ranking(d)
    st.caption('Words ranked by how much the citizen and LLM bands overlap (highest first). Bars: 95% CI. '
               'Colour: hedge words (H2a), extreme words (H2a), other.')
    base = alt.Chart(r).encode(y=alt.Y('term:N', sort=list(r.term), title=None))
    cat = alt.Color('category:N', title='Word type', scale=alt.Scale(domain=['hedge', 'extreme', 'other'],
                                                                      range=[ORANGE, BLUE, GREY]))
    chart = base.mark_rule(strokeWidth=2).encode(x=alt.X('ci_lo:Q', title='Overlap (Jaccard)', scale=alt.Scale(domain=[0, 1])),
                                                 x2='ci_hi:Q', color=cat) + \
        base.mark_point(filled=True, size=70).encode(
            x='jaccard:Q', color=cat,
            tooltip=['rank:Q', 'term:N', alt.Tooltip('jaccard:Q', format='.2f'), alt.Tooltip('ci_lo:Q', format='.2f'),
                     alt.Tooltip('ci_hi:Q', format='.2f'), alt.Tooltip('centre_shift:Q', format='+.1f', title='centre shift')])
    st.altair_chart(chart.properties(height=480), width='stretch')
    st.dataframe(r.rename(columns={'jaccard': 'overlap', 'ci_lo': 'CI low', 'ci_hi': 'CI high',
                                   'centre_shift': 'centre shift (pp)', 'category': 'word type'}),
                 hide_index=True, width='stretch')
