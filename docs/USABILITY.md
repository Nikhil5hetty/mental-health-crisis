# Usability: heuristic self-evaluation of the word-meaning lab (WP6)

**Status: DRAFT.** Written by Claude Code (WP6). It stays a draft until Nikhil and Rajeshwar have reviewed it
(PM ruling on PR #32).

Method: a walk-through of both pages of `app/word_lab.py` (v. WP6, 2026-11-24) against Nielsen's ten
usability heuristics (Nielsen 1994, *Enhancing the explanatory power of usability heuristics*, CHI '94). One line
per heuristic: what the app does, and what remains open. Severity of open issues: 0 = none, 1 = cosmetic,
2 = minor, 3 = major (Nielsen's scale). This is a self-evaluation by the developers, not a user study.

| # | Heuristic | How the app meets it | Open issue (severity) |
|---|---|---|---|
| 1 | Visibility of system status | Every control updates the chart and table at once. The simulator banner states the current verdict ("Decision flips" or "Same decision"), and the page says when RQ3 results are absent. | The slider value only shows on the handle; it is not repeated in the verdict text (1). |
| 2 | Match between system and the real world | Plain-language labels: "crisis degree", "quality of the bot's reply", "Suggest a helpline (143; 147)", "no crisis / possible / likely / clear". Technical terms (Jaccard, IT2) appear only next to a plain gloss. | "Membership" and "fit" are still jargon for lay users (2). |
| 3 | User control and freedom | No irreversible actions. Toggles and the slider can be reset freely, and pages switch without losing anything that matters. | No "reset to default" button (1). |
| 4 | Consistency and standards | Same colours throughout (citizens blue, LLMs orange), the same action codes A0–A3 on both pages, and standard Streamlit widgets and navigation. | Page 1 calls it "membership", page 2 calls it "fit" (1). |
| 5 | Error prevention | Inputs are constrained: a slider limited to 0–100, a fixed list of reply words and a fixed list of terms. The logic rejects out-of-range values (unit-tested). | none (0) |
| 6 | Recognition rather than recall | Options are visible as drop-down lists with short explanations of the reply words. The action legend is repeated under the table, and a disclaimer is always visible in the sidebar. | The ranking tab does not link back to the single-word view (2). |
| 7 | Flexibility and efficiency of use | Toggles let expert users add model and group lines; novices get the default view. Tooltips give exact numbers on hover, and a results-folder override serves power users. | No keyboard shortcuts or shareable URL state for a chosen term or slider value (2). |
| 8 | Aesthetic and minimalist design | Each page has one main chart or table, three headline numbers and details in expanders ("How to read this", "Per-model details"). | Page 1 legends take horizontal space on narrow screens (1). |
| 9 | Help users recognise, diagnose and recover from errors | Missing result files produce a message naming each missing file and the exact command that produces it (tested). Excluded models are explained in a warning. | none (0) |
| 10 | Help and documentation | A "How to read this" box on both pages, ⓘ help on every control and metric, and `app/README.md` with an installation manual, a user manual with screenshots and known limitations. | No in-app link to the README (1). |

**Summary.** No major (severity 3) issues found. The main follow-ups are plainer wording for "membership/fit"
(H2), linking ranking rows to the single-word view (H6), and shareable URL state (H7). A small think-aloud test
with 3–5 non-expert users is recommended before any presentation to stakeholders.
