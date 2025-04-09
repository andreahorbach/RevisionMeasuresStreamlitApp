import streamlit as st
import pandas as pd
from io import BytesIO
from apply_measures import apply_measures

st.title("Revision Measures")

if "revision_measures" not in st.session_state:
    st.session_state.revision_measures = {
        "all": False,
        "length_diff": False,
        "length_diff_tokens": False,
        "levenshtein": False,
        "levenshtein_tokens": False,
        "levenshtein_pos": False,
        "gst": False,
        "gst_tokens": False,
        "gst_pos": False,
        "lcs": False,
        "lcs_tokens": False,
        "lcs_pos": False,
        "sbert": False
    }

measures_name_mapping = {
        "length_diff":"length_diff_chars",
        "length_diff_tokens": "length_diff_tokens",
        "levenshtein":  "levenshtein_chars",
        "levenshtein_tokens": "levenshtein_tokens",
        "levenshtein_pos": "levenshtein_pos",
        "gst": "gst_chars",
        "gst_tokens": "gst_tokens",
        "gst_pos": "gst_pos",
        "lcs":  "lcs_chars",
        "lcs_tokens": "lcs_tokens",
        "lcs_pos": "lcs_pos",
        "sbert": "sbert_similarity"
}

# --- Step 1: Configuration ---

st.subheader("Configure Parameters")

# Language select
language = st.selectbox("Language for processing", options=["Deutsch", "English"],
help="Select the language of the text in your data")

# Revision Measures in Grid
st.markdown("**Revision Measures**")

# "All"-Checkbox
all_selected = st.checkbox(
    "All revision measures",
    value=st.session_state.revision_measures["all"],
    help="Selects or deselects all revision measures below"
)

# Wenn All geändert wurde, synchronisiere alle anderen
if all_selected != st.session_state.revision_measures["all"]:
    st.session_state.revision_measures["all"] = all_selected
    for key in st.session_state.revision_measures:
        if key != "all":
            st.session_state.revision_measures[key] = all_selected

# Checkbox-Labels & Tooltips
checkboxes = {
    "length_diff": ("Length Difference", "Length difference in characters between the first and second text. Can be negative if the second text is shorter than the first"),
    "length_diff_tokens": ("Length Difference (tokens)", "Length difference as before but in tokens."),
    "levenshtein": ("Levenshtein", "Levenshtein distance or edit distance: the numer of inserts, deletions or changes on character level to get from the first to the second text."),
    "levenshtein_tokens": ("Levenshtein (tokens)", "Levenshtein distance based on token sequences"),
    "levenshtein_pos": ("Levenshtein (POS)", "Levenshtein distance based on POS tags"),
    "gst": ("GST", "Greedy String Tiling on characters"),
    "gst_tokens": ("GST (tokens)", "Greedy String Tiling on tokens"),
    "gst_pos": ("GST (POS)", "Greedy String Tiling on POS tags"),
    "lcs": ("LCS", "Longest Common Subsequence on characters"),
    "lcs_tokens": ("LCS (tokens)", "LCS on tokens"),
    "lcs_pos": ("LCS (POS)", "LCS on POS tags"),
    "sbert": ("SBERT", "Semantic similarity")
}

# --- Zeile 1: Length Difference (2 Checkboxen) ---
row1 = st.columns(3)
with row1[0]:
    st.session_state.revision_measures["length_diff"] = st.checkbox(
        checkboxes["length_diff"][0],
        value=st.session_state.revision_measures["length_diff"],
        help=checkboxes["length_diff"][1]
    )
with row1[1]:
    st.session_state.revision_measures["length_diff_tokens"] = st.checkbox(
        checkboxes["length_diff_tokens"][0],
        value=st.session_state.revision_measures["length_diff_tokens"],
        help=checkboxes["length_diff_tokens"][1]
    )
# row1[2] bleibt leer

# --- Zeile 2: Levenshtein ---
row2 = st.columns(3)
for i, key in enumerate(["levenshtein", "levenshtein_tokens", "levenshtein_pos"]):
    with row2[i]:
        st.session_state.revision_measures[key] = st.checkbox(
            checkboxes[key][0],
            value=st.session_state.revision_measures[key],
            help=checkboxes[key][1]
        )

# --- Zeile 3: GST ---
row3 = st.columns(3)
for i, key in enumerate(["gst", "gst_tokens", "gst_pos"]):
    with row3[i]:
        st.session_state.revision_measures[key] = st.checkbox(
            checkboxes[key][0],
            value=st.session_state.revision_measures[key],
            help=checkboxes[key][1]
        )

# --- Zeile 4: LCS ---
row4 = st.columns(3)
for i, key in enumerate(["lcs", "lcs_tokens", "lcs_pos"]):
    with row4[i]:
        st.session_state.revision_measures[key] = st.checkbox(
            checkboxes[key][0],
            value=st.session_state.revision_measures[key],
            help=checkboxes[key][1]
        )

row5 = st.columns(3)
with row5[0]:
    st.session_state.revision_measures["sbert"] = st.checkbox(
        checkboxes["sbert"][0],
        value=st.session_state.revision_measures["sbert"],
        help=checkboxes["sbert"][1]
    )

# --- All-Checkbox aktualisieren, wenn nötig ---
subkeys = [k for k in st.session_state.revision_measures if k != "all"]
if all(st.session_state.revision_measures[k] for k in subkeys):
    st.session_state.revision_measures["all"] = True
elif any(not st.session_state.revision_measures[k] for k in subkeys):
    st.session_state.revision_measures["all"] = False


# Raw vs Normalized (horizontal)

st.markdown("**Value Type**")

value_type = st.radio(
    label="Choose between raw values and values normalized by essay length",
    options=["Raw Values", "Normalized Values"],
    horizontal=True,
)

# --- Step 2: Upload File ---

st.subheader("Upload File")
uploaded_file = st.file_uploader("Upload an Excel or CSV file. For a csv file, the delimiter has to be a comma. The input file needs to have the two columns text1 and text2. Any other columns will be ignored.", type=["xlsx", "csv"])

# --- Step 3: Run Button ---

run_disabled = uploaded_file is None
run_clicked = st.button("Run", disabled=run_disabled)

# --- Step 4: Processing, Preview & Download ---

if run_clicked and uploaded_file:

    lang = "de" if language == "Deutsch" else "en"
    abs_or_norm = "abs" if value_type == "Raw Values" else "norm"

    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=";")
    else:
        df = pd.read_excel(uploaded_file)

    #drop rows where one text is empty
    df = df.loc[(df["text1"].notna()) & (df["text2"].notna())]

    checked_measures = [measure for measure, checked in st.session_state.revision_measures.items() if checked and measure != "all"]
    checked_measures = [measures_name_mapping[measure] for measure in checked_measures]

    processed_df = apply_measures(df=df,
                   lang=lang,
                   column_text1="text1",
                   column_text2="text2",
                   abs_or_norm=abs_or_norm,
                   normalize_by="text1",
                   outfilename="results",
                   measures=checked_measures)


    # Preview
    st.subheader("Preview")
    st.dataframe(processed_df.head(10))

    # Prepare Excel for download
    def to_excel(dataframe):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            dataframe.to_excel(writer, index=False, sheet_name='Processed')
        output.seek(0)
        return output

    # Download button
    st.subheader("Download")
    st.download_button(
        label="Download processed Excel file",
        data=to_excel(processed_df),
        file_name="processed_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
