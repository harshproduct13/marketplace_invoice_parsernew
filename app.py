import streamlit as st
from openai import OpenAI
import sqlite3
import pandas as pd
import json
import re
import io
import os
from PIL import Image
import base64
import traceback

# ---------------- CONFIG ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("⚠️ Please set OPENAI_API_KEY in Streamlit secrets or environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DB_PATH = "invoices.db"

# ---------------- DATABASE SETUP ----------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS invoice_line_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    marketplace_name TEXT,
    invoice_type TEXT,
    invoice_date TEXT,
    place_of_supply TEXT,
    gstin TEXT,
    service_description TEXT,
    net_taxable_value REAL,
    total_IGST_amount REAL,
    total_CGST_amount REAL,
    total_SGST_amount REAL,
    total_amount REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")
conn.commit()

# ---------------- PROMPTS ----------------

FLIPKART_PROMPT = """
You are analyzing an image of a Flipkart invoice or credit note.

Flipkart invoices are horizontally structured: each row represents one service or fee with its tax values.
Extract all line items and return them as a JSON array.

Each JSON object must have:
[
  {
    "Marketplace Name": "Flipkart",
    "Types of Invoice": "Tax Invoice" | "Credit Note" | "Commercial Credit Note",
    "Date of Invoice/Credit Note": "DD-MM-YYYY",
    "Place of Supply": "STATE, IN-XX",
    "GSTIN": "string or null",
    "Service Description": "string",
    "Net Taxable Value": number,
    "total_IGST_amount": number or null,
    "total_CGST_amount": number or null,
    "total_SGST_amount": number or null,
    "total_amount": number
  }
]

Rules:
- If IGST is applied, set CGST and SGST to null.
- If CGST and SGST are applied, set IGST to null.
- If it's a Credit Note, make all numeric values negative.
- Return valid JSON only, with no markdown or explanation.
"""

AMAZON_PROMPT = """
You are analyzing an image of an Amazon invoice or credit note.

Amazon invoices are often vertically structured. Each service is followed by one or more tax rows (SGST, CGST, or IGST).
Combine those into a single JSON object per service, summing the total correctly.

For each service group:
- Identify the service name (e.g., Shipping Fee, Pick & Pack Fee, FBA Fee).
- Add up any tax amounts that follow it.
- Compute:
  Total Amount = Net Taxable Value + sum of tax amounts.
- If the document is a credit note, all values should be negative.

Return all services as an array of JSON objects with this exact schema:
[
  {
    "Marketplace Name": "Amazon",
    "Types of Invoice": "Tax Invoice" | "Credit Note",
    "Date of Invoice/Credit Note": "DD-MM-YYYY",
    "Place of Supply": "STATE, IN-XX",
    "GSTIN": "string or null",
    "Service Description": "string",
    "Net Taxable Value": number,
    "total_IGST_amount": number or null,
    "total_CGST_amount": number or null,
    "total_SGST_amount": number or null,
    "total_amount": number
  }
]

Rules:
- Group SGST and CGST with their parent service.
- If only IGST exists, set IGST field and leave others null.
- If SGST and CGST exist, fill both and set IGST to null.
- Do not include separate tax lines as services.
- Return strict JSON only, no explanations.
"""

# ---------------- HELPERS ----------------

def extract_json(text):
    """Extract JSON array from model output."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, re.S)
        if match:
            try:
                cleaned = re.sub(r",\s*([}\]])", r"\1", match.group(0))
                return json.loads(cleaned)
            except:
                pass
    return None


def sanitize_number(value):
    """Convert numbers safely."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    s = re.sub(r"[^\d.\-]", "", str(value))
    if not s:
        return None
    try:
        return float(s)
    except:
        return None


def insert_rows(rows):
    """Insert parsed rows into DB."""
    for r in rows:
        cur.execute("""
            INSERT INTO invoice_line_items
            (marketplace_name, invoice_type, invoice_date, place_of_supply, gstin,
             service_description, net_taxable_value, total_IGST_amount, total_CGST_amount,
             total_SGST_amount, total_amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r.get("Marketplace Name"),
            r.get("Types of Invoice"),
            r.get("Date of Invoice/Credit Note"),
            r.get("Place of Supply"),
            r.get("GSTIN"),
            r.get("Service Description"),
            sanitize_number(r.get("Net Taxable Value")),
            sanitize_number(r.get("total_IGST_amount")),
            sanitize_number(r.get("total_CGST_amount")),
            sanitize_number(r.get("total_SGST_amount")),
            sanitize_number(r.get("total_amount")),
        ))
    conn.commit()


def delete_row(row_id):
    """Delete a record from the table."""
    cur.execute("DELETE FROM invoice_line_items WHERE id = ?", (row_id,))
    conn.commit()


def fetch_all_rows():
    """Fetch all invoices from DB."""
    return pd.read_sql_query(
        "SELECT id, marketplace_name, invoice_type, invoice_date, place_of_supply, gstin, "
        "service_description, net_taxable_value, total_IGST_amount, total_CGST_amount, total_SGST_amount, total_amount "
        "FROM invoice_line_items ORDER BY created_at DESC",
        conn
    )


def image_to_base64(image: Image.Image) -> str:
    """Convert image to base64 for API call."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def detect_marketplace_from_text(text):
    """Identify if invoice is from Amazon or Flipkart."""
    t = text.lower()
    if "amazon" in t:
        return "Amazon"
    elif "flipkart" in t:
        return "Flipkart"
    else:
        return "Unknown"


def call_openai_vision(image: Image.Image, prompt):
    """Send image + prompt to GPT-4o Vision."""
    img_b64 = image_to_base64(image)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt.strip()},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}
        ],
        temperature=0,
        max_tokens=1800
    )
    return response.choices[0].message.content

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Marketplace Invoice Parser", layout="wide")
st.title("🧾 Marketplace Invoice Parser (Amazon & Flipkart)")

uploaded_file = st.file_uploader("Upload Invoice Image", type=["jpg", "jpeg", "png"])
parse_button = st.button("Parse & Save Data")

if parse_button:
    if not uploaded_file:
        st.warning("Please upload an image first.")
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            with st.spinner("🔍 Detecting marketplace..."):
                text_preview = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "Identify if this invoice is from Amazon or Flipkart. Return only the name."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(image)}"}}
                    ]}],
                    temperature=0,
                    max_tokens=50
                )
                detected_source = text_preview.choices[0].message.content.strip()

            prompt = AMAZON_PROMPT if "amazon" in detected_source.lower() else FLIPKART_PROMPT

            with st.spinner(f"🧠 Parsing {detected_source} invoice..."):
                llm_output = call_openai_vision(image, prompt)
            parsed = extract_json(llm_output)

            if not parsed:
                st.error("❌ Could not parse valid JSON. Try again or check the image.")
            else:
                insert_rows(parsed)
                st.success(f"✅ Parsed and saved {len(parsed)} line items successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
            st.error(traceback.format_exc())

# ---------------- TABLE VIEW ----------------
st.markdown("---")
st.subheader("📊 Stored Invoice Line Items")

df = fetch_all_rows()
if df.empty:
    st.info("No records yet. Upload an invoice to begin.")
else:
    col_table, col_buttons = st.columns([18, 1])

    with col_table:
        st.download_button(
            label="⬇️ Download as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="invoice_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(df.drop(columns=["id"]), use_container_width=True, hide_index=True)

    with col_buttons:
        st.markdown("<div style='margin-top: 35px;'></div>", unsafe_allow_html=True)
        for _, row in df.iterrows():
            btn_key = f"delete_{row['id']}"
            if st.button("🗑️", key=btn_key):
                delete_row(row["id"])
                st.rerun()
