"""Visual identity for the Credit Risk Scorer.

Design language: light institutional fintech. Warm ivory paper, deep navy ink,
a Spectral serif for headings paired with Inter for body — restrained, the way
a lending product or a bank's risk tool would look. The only saturated colour
is the approve/decline decision itself.
"""

import streamlit as st

INK = "#16211c"
NAVY = "#1e6f52"
MUTED = "#5c6678"
PAPER = "#f9f7f2"
APPROVE = "#1f7a52"
DECLINE = "#b4232a"
LINE = "#ddd7c8"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="st-"], [data-testid="stMarkdownContainer"], input, button, select {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
[data-testid="stAppViewContainer"] { background: #f9f7f2; }

h1, h2, h3 {
  font-family: 'Spectral', Georgia, serif !important;
  color: #16211c !important;
  letter-spacing: -0.01em;
}
h1 { font-weight: 700 !important; font-size: 2.5rem !important; line-height: 1.1; }
h2 { font-weight: 600 !important; }
h3 { font-weight: 600 !important; }

/* metric cards — white, hairline border, soft lift */
[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid #e7e3d8;
  border-radius: 10px;
  padding: 16px 20px;
  box-shadow: 0 1px 2px rgba(20,34,61,0.05);
}
[data-testid="stMetricLabel"] { color: #5c6678 !important; }
[data-testid="stMetricValue"] { color: #16211c !important; font-weight: 600; }

[data-testid="stSidebar"] {
  background: #f3f0e8;
  border-right: 1px solid #e1dccf;
}

.stButton button {
  border-radius: 6px;
  border: 1px solid #c9c2b1;
  background: #ffffff;
  color: #1e6f52;
  font-weight: 500;
}
.stButton button:hover { border-color: #1e6f52; background: #f3f0e8; }

[data-testid="stTabs"] button[role="tab"] {
  font-weight: 600;
  color: #5c6678;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: #1e6f52; }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

div[data-testid="stExpander"] {
  border: 1px solid #e7e3d8;
  border-radius: 8px;
  background: #ffffff;
}

/* an eyebrow / kicker above the masthead */
.eyebrow {
  text-transform: uppercase; letter-spacing: 0.18em;
  font-size: 0.72rem; color: #c8851f; font-weight: 600;
  margin-bottom: 2px;
}
/* a thin rule under the masthead, the way print sets a headline */
.masthead-rule { height: 2px; background: #1e6f52; width: 64px; margin: 6px 0 4px; }

.dl-step {
  background: #ffffff;
  border: 1px solid #e7e3d8;
  border-radius: 10px;
  padding: 18px;
  height: 100%;
}
.dl-step b { color: #1e6f52; font-family: 'Spectral', serif; font-size: 1.02rem; }
.dl-step .n {
  display: inline-block; width: 28px; height: 28px; line-height: 28px;
  text-align: center; border-radius: 50%;
  border: 1.5px solid #1e6f52; color: #1e6f52;
  font-weight: 600; margin-bottom: 10px; font-size: 0.85rem;
  font-family: 'Spectral', serif;
}

.reason {
  padding: 10px 16px; margin: 7px 0;
  border-left: 3px solid; border-radius: 0 8px 8px 0;
  background: #ffffff; border-top: 1px solid #efe9dc;
  border-right: 1px solid #efe9dc; border-bottom: 1px solid #efe9dc;
  font-size: 0.95rem; color: #2b3447;
}
.reason.neg { border-left-color: #b4232a; }
.reason.pos { border-left-color: #1f7a52; }
.reason.tip { border-left-color: #1e6f52; }

/* decision — a stamped verdict, not a glowing pill */
.pill {
  display: inline-block; padding: 6px 18px; border-radius: 4px;
  font-family: 'Spectral', serif; font-weight: 700;
  letter-spacing: 0.08em; font-size: 1.1rem; text-transform: uppercase;
}
.pill.ok { background: #e7f1ea; color: #1f7a52; border: 1px solid #1f7a52; }
.pill.no { background: #f6e7e7; color: #b4232a; border: 1px solid #b4232a; }

table { font-size: 0.92rem; }
a { color: #1e6f52; }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def eyebrow(text: str):
    st.markdown(
        f'<p class="eyebrow">{text}</p><div class="masthead-rule"></div>',
        unsafe_allow_html=True,
    )


def verdict_pill(approved: bool, label_ok: str = "Approved",
                 label_no: str = "Declined"):
    cls, label = ("ok", label_ok) if approved else ("no", label_no)
    st.markdown(f'<span class="pill {cls}">{label}</span>', unsafe_allow_html=True)


def reason(text: str, kind: str):
    st.markdown(f'<div class="reason {kind}">{text}</div>', unsafe_allow_html=True)


def style_fig(fig):
    """Restyle a matplotlib/SHAP figure for the light paper theme."""
    import matplotlib.pyplot as plt

    fig.patch.set_facecolor(PAPER)
    for ax in fig.axes:
        ax.set_facecolor(PAPER)
        ax.tick_params(colors=INK, labelcolor=INK)
        for spine in ax.spines.values():
            spine.set_color("#cfc8b8")
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        ax.title.set_color(INK)
    return fig


def step(n, title, body):
    st.markdown(
        f'<div class="dl-step"><span class="n">{n}</span><br/>'
        f'<b>{title}</b><br/><span style="color:#5c6678;font-size:0.92rem">{body}</span></div>',
        unsafe_allow_html=True,
    )


def footer(repo: str):
    st.divider()
    st.markdown(
        f'<p style="color:#7a8090;font-size:0.85rem">Built by '
        f'<a href="https://drishtantleuva.github.io" target="_blank">'
        f'<b style="font-family:\'Fraunces\',Georgia,serif">Drishtant Leuva</b></a> '
        f'— Data Scientist · Risk &amp; Explainable AI &nbsp;·&nbsp; '
        f'<a href="https://github.com/drishtantleuva/{repo}" target="_blank">Source on GitHub</a> '
        f'&nbsp;·&nbsp; <a href="https://www.linkedin.com/in/drishtant-leuva/" '
        f'target="_blank">LinkedIn</a></p>',
        unsafe_allow_html=True,
    )
