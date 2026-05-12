import re
import shlex
from typing import List

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz 

# ----------------------------
# App Configuration
# ----------------------------
st.set_page_config(
    page_title="Tradewinds Marketplace Search",
    page_icon="🔎",
    layout="wide",
)

# ----------------------------
# Expected Columns & Curated Options (from the dataset)
# ----------------------------
EXPECTED_COLUMNS = [
    "Vendor",
    "Submission Title",
    "Strategic Focus Area",
    "UEI",
    "FedRAMP Status",
    "Business Size",
    "Contractor Type",
    "Related Keywords",
    "POC",
    "Abstract",
    "Video Transcript",
]

DEFAULT_FILE = "Tradewinds Marketplace Portal.xlsx"

FEDRAMP_OPTIONS = ["", "Not Applicable", "No", "Process", "Yes"]

SFA_OPTIONS = [
    "Application of AI/ML Scaffolding",
    "Assessment and Compliance Solutions",
    "Assuring cybersecurity",
    "Assuring Reliable Data Sources",
    "Biomedical and Human Performance Solutions",
    "Developing a digital-age workforce",
    "Discovering blue sky/other technology applications",
    "Enhancing Lethality",
    "Frontier Artificial Intelligence Solutions",
    "Implementing predictive maintenance and supply",
    "Improving situational awareness and decision-making",
    "Increasing autonomy and mobility of DoD systems",
    "Increasing safety of operating equipment",
    "Research Solutions and Services",
    "Special Topic - AI Detection",
    "Special Topic - Assurance",
    "Special Topic - Flex Fuel (Original)",
    "Special Topic - Flex Fuel (ReRun)",
    "Special Topic - Geothermal",
    "Special Topic - GIDE",
    "Streamlining business processes",
]

# ----------------------------
# Helpers
# ----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim and normalize column names."""
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_excel(path_or_buffer) -> pd.DataFrame:
    """Load Excel and ensure strings + missing handled."""
    df = pd.read_excel(path_or_buffer, sheet_name=0, dtype=str)
    df = normalize_columns(df)
    # Ensure all expected columns exist (missing will be added as empty strings)
    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    # Normalize string values (strip trailing spaces etc.)
    for col in EXPECTED_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df

def parse_query(q: str) -> List[str]:
    """
    Parse a query string into tokens, supporting quoted phrases.
    Example: 'zero trust "access control"' -> ['zero', 'cess control']
    """
    q = q.strip()
    if not q:
        return []
    try:
        tokens = shlex.split(q)
    except Exception:
        tokens = q.split()
    # Deduplicate while preserving order
    seen, out = set(), []
    for t in tokens:
        if t and t.lower() not in seen:
            out.append(t)
            seen.add(t.lower())
    return out

def contains_any_exact(df: pd.DataFrame, cols: List[str], term: str) -> pd.Series:
    """Row-wise boolean: does any selected column contain 'term' (case-insensitive, exact substring)?"""
    return df[cols].apply(lambda col: col.str.contains(re.escape(term), case=False, na=False)).any(axis=1)

def build_keyword_mask(
    df: pd.DataFrame,
    cols_to_search: List[str],
    query: str,
    mode: str,
    exact_phrase: bool,
    fuzzy_enabled: bool,
    fuzzy_threshold: int,
    fuzzy_method: str,
) -> pd.Series:
    """
    Build a boolean mask for keyword query using exact or fuzzy matching.
    - mode: 'AND' or 'OR'
    - exact_phrase: True -> treat the entire query as a single phrase
    - fuzzy_enabled: use RapidFuzz with threshold and method
    - fuzzy_method: 'partial', 'token_set', 'token_sort'
    """
    if not query.strip() or not cols_to_search:
        return pd.Series(True, index=df.index)

    # Combine selected columns for faster fuzzy checks
    # Lowercase once to avoid repeated lower() calls
    search_texts = (
        df[cols_to_search]
        .apply(lambda r: " ".join([str(x) for x in r.values]), axis=1)
        .str.lower()
    )

    if fuzzy_enabled:
        def score(text: str, term: str) -> int:
            term = term.lower().strip()
            if not term:
                return 100
            if fuzzy_method == "partial":
                return fuzz.partial_ratio(term, text)
            elif fuzzy_method == "token_set":
                return fuzz.token_set_ratio(term, text)
            else:  
                return fuzz.token_sort_ratio(term, text)

        if exact_phrase:
            phrase = query.strip()
            return search_texts.apply(lambda t: score(t, phrase) >= fuzzy_threshold)

        tokens = parse_query(query)
        if not tokens:
            return pd.Series(True, index=df.index)

        if mode == "AND":
            mask = pd.Series(True, index=df.index)
            for t in tokens:
                mask &= search_texts.apply(lambda txt: score(txt, t) >= fuzzy_threshold)
            return mask
        else:  # OR mode
            # For OR, any token meeting threshold is enough
            def any_token_matches(txt: str) -> bool:
                return any(score(txt, t) >= fuzzy_threshold for t in tokens)
            return search_texts.apply(any_token_matches)

    # ---- Exact (non-fuzzy) path ----
    if exact_phrase:
        phrase_mask = contains_any_exact(df, cols_to_search, query.strip())
        return phrase_mask

    tokens = parse_query(query)
    if not tokens:
        return pd.Series(True, index=df.index)

    if mode == "AND":
        mask = pd.Series(True, index=df.index)
        for t in tokens:
            mask &= contains_any_exact(df, cols_to_search, t)
        return mask
    else:
        masks = [contains_any_exact(df, cols_to_search, t) for t in tokens]
        return pd.concat(masks, axis=1).any(axis=1)

def highlight_text(text: str, tokens: List[str]) -> str:
    """Bold matched tokens in text using markdown. Case-insensitive exact highlights only."""

    if not text or not tokens:
        return text
    highlighted = text
    for t in tokens:
        if not t:
            continue
        pattern = re.compile(re.escape(t), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"**{m.group(0)}**", highlighted)
    return highlighted

def valid_columns(df: pd.DataFrame, expected=EXPECTED_COLUMNS):
    missing = [c for c in expected if c not in df.columns]
    return len(missing) == 0, missing

# ----------------------------
# Sidebar - Data Source
# ----------------------------
try:
    df = load_excel("Tradewinds Marketplace Portal.xlsx") # Only my data set will be used, option to upload one's own spreadsheet is unavailable
except Exception as e:
    st.error("Could not load the data file 'Tradewinds Marketplace Portal.xlsx'. "
             "Make sure it is in the same folder as app.py.\n\n" + str(e))
    st.stop()


ok, missing = valid_columns(df)
if not ok:
    st.warning(
        "The file is missing expected columns:\n- " + "\n- ".join(missing) +
        "\n\nMake sure your Excel has these columns:\n" + ", ".join(EXPECTED_COLUMNS)
    )

for col in EXPECTED_COLUMNS:
    df[col] = df[col].astype(str).str.strip()

# ----------------------------
# Sidebar - Search & Filters
# ----------------------------
st.sidebar.header("Search")

query = st.sidebar.text_input(
    "Keywords or \"exact phrase\"",
    placeholder='e.g., zero trust "identity management"',
)

search_columns_default = [
    "Submission Title", "Vendor", "Related Keywords", "Abstract", "Video Transcript"
]
search_columns = st.sidebar.multiselect(
    "Search in columns",
    options=EXPECTED_COLUMNS,
    default=[c for c in search_columns_default if c in df.columns],
)

match_mode = st.sidebar.radio("Match mode", ["AND", "OR"], index=0, horizontal=True)
exact_phrase = st.sidebar.checkbox("Exact phrase (ignore AND/OR)", value=False)

# Fuzzy controls to allow for similar results instead of exact matches
st.sidebar.subheader("Fuzzy Matching")
fuzzy_enabled = st.sidebar.checkbox("Enable fuzzy match (typo tolerance)", value=True)
fuzzy_method = st.sidebar.radio(
    "Fuzzy method",
    options=["partial", "token_set", "token_sort"],
    index=0,
    help=(
        "partial: best for substrings/typos\n"
        "token_set: ignores word order, good for phrases with extra words\n"
        "token_sort: sorts words before comparing"
    ),
    horizontal=True,
)
fuzzy_threshold = st.sidebar.slider(
    "Fuzzy threshold",
    min_value=40, max_value=95, value=80, step=1,
    help="Higher is stricter (exact ~90+). 60–75 allows moderate typos."
)

st.sidebar.header("Filters")
vendor_filter = st.sidebar.text_input("Vendor contains", placeholder="partial match")
uei_filter = st.sidebar.text_input("UEI contains", placeholder="partial match")

# Use my curated options for these filters
selected_focus = st.sidebar.multiselect(
    "Strategic Focus Area",
    options=SFA_OPTIONS,
)
selected_fedramp = st.sidebar.multiselect(
    "FedRAMP Status",
    options=FEDRAMP_OPTIONS,
)

size_values = sorted([v for v in df["Business Size"].unique() if v])
selected_size = st.sidebar.multiselect("Business Size", options=size_values)

ctype_values = sorted([v for v in df["Contractor Type"].unique() if v])
selected_ctype = st.sidebar.multiselect("Contractor Type", options=ctype_values)

limit_rows = st.sidebar.number_input("Limit rows (for display)", min_value=10, max_value=5000, value=1000, step=50)

# ----------------------------
# Filtering Logic
# ----------------------------
kw_mask = build_keyword_mask(
    df=df,
    cols_to_search=search_columns,
    query=query,
    mode=match_mode,
    exact_phrase=exact_phrase,
    fuzzy_enabled=fuzzy_enabled,
    fuzzy_threshold=fuzzy_threshold,
    fuzzy_method=fuzzy_method,
)

col_mask = pd.Series(True, index=df.index)
if vendor_filter.strip():
    col_mask &= df["Vendor"].str.contains(re.escape(vendor_filter.strip()), case=False, na=False)
if uei_filter.strip():
    col_mask &= df["UEI"].str.contains(re.escape(uei_filter.strip()), case=False, na=False)
if selected_focus:
    # Note: We strip dataset values earlier; curated options are already stripped
    col_mask &= df["Strategic Focus Area"].isin(selected_focus)
if selected_fedramp:
    col_mask &= df["FedRAMP Status"].isin(selected_fedramp)
if selected_size:
    col_mask &= df["Business Size"].isin(selected_size)
if selected_ctype:
    col_mask &= df["Contractor Type"].isin(selected_ctype)

filtered = df[kw_mask & col_mask].copy()

# ----------------------------
# Main Layout
# ----------------------------
st.title("🔎 Tradewinds Marketplace Search")
st.caption("Search and filter vendors by keywords, phrases, and column designations. Fuzzy matching enabled for typo tolerance.")

left, right = st.columns([3, 2])

with left:
    st.subheader("Results")
    st.write(f"**{len(filtered):,}** matches (showing up to {limit_rows:,})")

    view_mode = st.radio("View", ["Table", "Cards"], index=0, horizontal=True)

    show_cols = st.multiselect(
        "Columns to display",
        options=EXPECTED_COLUMNS,
        default=[
            "Vendor", "Submission Title", "Strategic Focus Area", "FedRAMP Status",
            "Business Size", "Contractor Type", "UEI", "Related Keywords", "POC"
        ],
    )

    preview = filtered.head(limit_rows)
    tokens = parse_query(query) if not exact_phrase else [query.strip()] if query.strip() else []

    if view_mode == "Table":
        st.dataframe(preview[show_cols], use_container_width=True, height=450)
    else:
        for _, row in preview.iterrows():
            with st.container():
                title_text = row.get("Submission Title", "")
                st.markdown(f"### {highlight_text(title_text, tokens)}")
                meta_line = []
                if "Vendor" in show_cols and row["Vendor"]:
                    meta_line.append(f"**Vendor:** {highlight_text(row['Vendor'], tokens)}")
                if "Strategic Focus Area" in show_cols and row["Strategic Focus Area"]:
                    meta_line.append(f"**Focus:** {row['Strategic Focus Area']}")
                if "FedRAMP Status" in show_cols and row["FedRAMP Status"]:
                    meta_line.append(f"**FedRAMP:** {row['FedRAMP Status']}")
                if "Business Size" in show_cols and row["Business Size"]:
                    meta_line.append(f"**Size:** {row['Business Size']}")
                if "Contractor Type" in show_cols and row["Contractor Type"]:
                    meta_line.append(f"**Type:** {row['Contractor Type']}")
                if "UEI" in show_cols and row["UEI"]:
                    meta_line.append(f"**UEI:** {highlight_text(row['UEI'], tokens)}")
                st.write(" | ".join(meta_line))

                if "Related Keywords" in show_cols and row["Related Keywords"]:
                    st.caption(f"**Keywords:** {highlight_text(row['Related Keywords'], tokens)}")

                with st.expander("Abstract"):
                    st.markdown(highlight_text(row.get("Abstract", ""), tokens))
                with st.expander("Video Transcript"):
                    st.markdown(highlight_text(row.get("Video Transcript", ""), tokens))
                st.divider()

with right:
    st.subheader("Quick Summaries")
    if len(filtered):
        counts = (
            filtered["Strategic Focus Area"]
            .replace("", "Unspecified")
            .value_counts()
            .reset_index()
        )
        counts.columns = ["Strategic Focus Area", "Count"]
        st.bar_chart(data=counts, x="Strategic Focus Area", y="Count", use_container_width=True, height=300)

    st.subheader("Export")
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered results (CSV)",
        data=csv_bytes,
        file_name="tradewinds_filtered.csv",
        mime="text/csv",
    )

st.caption(
    "Tip: Use quotes for exact phrases (e.g., \"identity management\"). "
    "Fuzzy threshold ~60–75 is tolerant of typos; 85–90 is stricter. "
    "You can filter by Strategic Focus Area, FedRAMP Status, Business Size, and Contractor Type."
)
