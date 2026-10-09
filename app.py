from pathlib import Path
import sqlite3
from datetime import date

import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "data" / "waste_reuse.db"
st.set_page_config(page_title="Household Waste Reuse Intelligence", page_icon="♻️", layout="wide")

RULES = {
    "Plastic": {
        "Good": ("Reuse", "Reuse clean containers for storage if suitable."),
        "Worn": ("Repair or recycle", "Consider repair or repurposing; check local recycling rules."),
        "Damaged": ("Check local recycling rules", "Not all plastics are accepted by every recycling service."),
    },
    "Paper/Cardboard": {
        "Good": ("Reuse", "Reuse paper for notes or clean cardboard boxes for storage."),
        "Worn": ("Reuse or recycle", "Reuse where suitable or check local recycling options."),
        "Damaged": ("Recycle if accepted", "Keep paper and cardboard clean and dry where required locally."),
    },
    "Glass": {
        "Good": ("Reuse", "Reuse an intact, clean jar for suitable storage."),
        "Worn": ("Reuse or recycle", "Reuse if intact; otherwise check local glass collection rules."),
        "Damaged": ("Handle carefully", "Avoid direct handling of broken glass and follow safe-disposal guidance."),
    },
    "Metal": {
        "Good": ("Reuse", "Reuse the item if it is clean, intact, and safe."),
        "Worn": ("Reuse or recycle", "Consider repair or local metal recycling if accepted."),
        "Damaged": ("Repair or recycle", "Check whether safe repair is possible or use an appropriate collection service."),
    },
    "Textile": {
        "Good": ("Donate or reuse", "If clean and usable, consider donating or repurposing it."),
        "Worn": ("Reuse or textile collection", "Repurpose it where suitable or check local textile collection options."),
        "Damaged": ("Repair or textile collection", "Consider repairing it or checking local textile collection options."),
    },
    "Electronic": {
        "Good": ("Repair or donate responsibly", "Consider repair or donation through a trusted channel; protect personal data."),
        "Worn": ("Repair or e-waste collection", "Check repair options or use an authorised e-waste collection service."),
        "Damaged": ("E-waste collection", "Use an authorised e-waste collection point; do not put electronics in ordinary recycling."),
    },
    "Organic": {
        "Good": ("Compost where suitable", "Suitable fruit and vegetable scraps may be composted with an appropriate setup."),
        "Worn": ("Organic waste collection", "Follow local organic-waste collection or composting guidance."),
        "Damaged": ("Organic waste collection", "Follow local organic-waste collection or composting guidance."),
    },
    "Other": {
        "Good": ("Reuse if safe", "Consider a safe second use, donation, or repair."),
        "Worn": ("Reuse or check local guidance", "Consider repair, reuse, donation, or a suitable local collection service."),
        "Damaged": ("Check local guidance", "Identify the material and check local disposal guidance before discarding."),
    },
}

def connect_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def initialize_db():
    with connect_db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS item_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_date TEXT NOT NULL, item_name TEXT NOT NULL,
            material TEXT NOT NULL, condition TEXT NOT NULL,
            intended_action TEXT NOT NULL, notes TEXT)""")
        conn.commit()

def load_records():
    with connect_db() as conn:
        return pd.read_sql_query(
            "SELECT id, record_date, item_name, material, condition, intended_action, notes "
            "FROM item_records ORDER BY record_date DESC, id DESC", conn)

def save_record(record_date, item_name, material, condition, action, notes):
    with connect_db() as conn:
        conn.execute("""INSERT INTO item_records
            (record_date, item_name, material, condition, intended_action, notes)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (str(record_date), item_name.strip(), material, condition, action, notes.strip()))
        conn.commit()

def recommend(material, condition, item_name, notes):
    action, suggestion = RULES[material][condition]
    description = f"{item_name} {material} {condition} {notes}".lower()
    ideas = [
        "Reuse for storage or organisation",
        "Repair before replacing",
        "Donate if clean, safe, and usable",
        "Check local recycling or collection rules",
        "Use authorised e-waste collection for electronics",
        "Compost suitable organic scraps where permitted",
    ]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([description] + ideas)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return action, suggestion, ideas[int(scores.argmax())]

def seed_demo_data():
    demo = [
        ("2026-10-01", "Glass jar", "Glass", "Good", "Reuse", "Clean and intact"),
        ("2026-10-02", "Cardboard box", "Paper/Cardboard", "Good", "Reuse", "Could store books"),
        ("2026-10-03", "Old T-shirt", "Textile", "Worn", "Reuse or textile collection", "Fabric is usable"),
        ("2026-10-04", "Phone charger", "Electronic", "Damaged", "E-waste collection", "Not working"),
        ("2026-10-05", "Vegetable scraps", "Organic", "Good", "Compost where suitable", "Kitchen scraps"),
        ("2026-10-06", "Plastic container", "Plastic", "Good", "Reuse", "Clean container"),
    ]
    with connect_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM item_records").fetchone()[0]
        if count == 0:
            conn.executemany("""INSERT INTO item_records
                (record_date, item_name, material, condition, intended_action, notes)
                VALUES (?, ?, ?, ?, ?, ?)""", demo)
            conn.commit()

initialize_db()
st.title("♻️ Household Waste Reuse Intelligence")
st.caption("A prototype for recording household items and suggesting reuse, repair, donation, or responsible disposal.")

with st.sidebar:
    st.header("Demo data")
    if st.button("Load sample data"):
        seed_demo_data()
        st.success("Sample records loaded if the database was empty.")
    st.caption("Examples are fictional. Suggestions are general; check local rules.")

st.subheader("1. Add a household item")
with st.form("item_form", clear_on_submit=True):
    c1, c2 = st.columns(2)
    with c1:
        record_date = st.date_input("Date recorded", value=date.today())
        item_name = st.text_input("Item name", placeholder="e.g., glass jar")
        material = st.selectbox("Main material", list(RULES.keys()))
    with c2:
        condition = st.selectbox("Condition", ["Good", "Worn", "Damaged"])
        notes = st.text_input("Description (optional)", placeholder="e.g., clean, cracked, not working")
    submitted = st.form_submit_button("Get suggestion and save")
    if submitted:
        if not item_name.strip():
            st.error("Please enter an item name.")
        else:
            action, suggestion, idea = recommend(material, condition, item_name, notes)
            save_record(record_date, item_name, material, condition, action, notes)
            st.success(f"Suggested action: {action}")
            st.write(suggestion)
            st.caption(f"Related idea (TF-IDF text similarity): {idea}")
            st.rerun()

st.subheader("2. Explore an item without saving")
with st.expander("Try a recommendation"):
    test_name = st.text_input("Item to explore", value="old mobile phone")
    test_material = st.selectbox("Material to explore", list(RULES.keys()), index=5)
    test_condition = st.selectbox("Condition to explore", ["Good", "Worn", "Damaged"], index=2)
    test_notes = st.text_input("Extra details", value="not working")
    if st.button("Suggest options"):
        action, suggestion, idea = recommend(test_material, test_condition, test_name, test_notes)
        st.info(action)
        st.write(suggestion)
        st.caption(f"Related idea (TF-IDF text similarity): {idea}")

st.subheader("3. Item history")
df = load_records()
if df.empty:
    st.info("No items recorded yet. Add an item or load sample data.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Items recorded", len(df))
    c2.metric("Material categories", df["material"].nunique())
    c3.metric("Reuse/repair/donation candidates",
              int(df["intended_action"].str.contains("reuse|donate|repair", case=False, regex=True).sum()))
    st.bar_chart(df.groupby("material").size().rename("Items"))
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download item history as CSV", csv, "household_item_history.csv", "text/csv")

st.subheader("4. Optional image upload")
st.write("This starter version displays an uploaded photo but does not classify it automatically yet.")
uploaded_image = st.file_uploader("Upload a household item photo", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded item photo", use_container_width=True)

st.divider()
st.caption("Scope: suggestions are rule-based and text similarity uses TF-IDF. No trained computer-vision model or LLM is integrated yet.")
