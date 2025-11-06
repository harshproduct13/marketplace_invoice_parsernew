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
import asyncio
import traceback

# ---------------- CONFIG ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.warning("⚠️ Please set your OPENAI_API_KEY in environment variables or Streamlit secrets.")

client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DB_PATH = "invoices_v3.db"

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


def fetch_all_rows():
    return pd.read_sql_query(
        "SELECT id, marketplace_name, invoice_type, invoice_date, place_of_supply, gstin, "
        "service_description, net_taxable_value, total_IGST_amount, total_CGST_amount, total_SGST_amount, total_amount "
        "FROM invoice_line_items ORDER BY created_at DESC",
        conn
    )


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def detect_marketplace(image: Image.Image):
    img_b64 = image_to_base64(image)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "Identify if this invoice belongs to Amazon or Flipkart. Reply only 'Amazon' or 'Flipkart'."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]}],
        temperature=0,
        max_tokens=20
    )
    return response.choices[0].message.content.strip()


def parse_invoice_image(image: Image.Image):
    detected = detect_marketplace(image)
    prompt = AMAZON_PROMPT if "amazon" in detected.lower() else FLIPKART_PROMPT
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
        max_tokens=2000
    )
    return extract_json(response.choices[0].message.content)

# ---------------- ASYNC PROCESSING ----------------

async def process_image_async(file, semaphore, progress, i, total):
    async with semaphore:
        try:
            image = Image.open(file).convert("RGB")
            parsed = await asyncio.to_thread(parse_invoice_image, image)
            if parsed:
                insert_rows(parsed)
                st.toast(f"✅ {file.name} parsed ({len(parsed)} line items)", icon="✅")
            else:
                st.toast(f"⚠️ Could not parse {file.name}", icon="⚠️")
        except Exception as e:
            st.toast(f"❌ Error with {file.name}: {e}", icon="❌")
            traceback.print_exc()
        finally:
            progress.progress((i + 1) / total)


async def process_all_images_async(files):
    total = len(files)
    progress = st.progress(0)
    semaphore = asyncio.Semaphore(5)

    tasks = []
    for i, file in enumerate(files):
        task = asyncio.create_task(process_image_async(file, semaphore, progress, i, total))
        tasks.append(task)

    await asyncio.gather(*tasks)
    progress.progress(1.0)
    st.success("🎉 All invoices processed successfully!")

# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="Marketplace Invoice Parser", layout="wide")
st.title("⚡ Async Marketplace Invoice Parser (Amazon + Flipkart)")

uploaded_files = st.file_uploader("Upload up to 10 invoice images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
parse_button = st.button("🚀 Parse & Save Data (Async)")

if parse_button:
    if not uploaded_files:
        st.warning("Please upload at least one image.")
    else:
        if len(uploaded_files) > 10:
            uploaded_files = uploaded_files[:10]
            st.info("Only the first 10 images will be processed.")
        asyncio.run(process_all_images_async(uploaded_files))
        st.rerun()

# ---------------- TABLE VIEW ----------------
st.markdown("---")
st.subheader("📊 Stored Invoice Line Items")

df = fetch_all_rows()
if df.empty:
    st.info("No records yet. Upload some invoices to begin.")
else:
    st.download_button(
        label="⬇️ Download as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="invoice_data.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.dataframe(df.drop(columns=["id"]), use_container_width=True, hide_index=True)

    # --- Delete All Button Section ---
    st.markdown("---")
    st.markdown("### ⚙️ Actions")
    if st.button("🗑️ Delete All Entries", type="secondary", use_container_width=True):
        st.warning("⚠️ This will permanently delete all invoice entries.")
        confirm = st.checkbox("I understand, delete all data permanently")
        if confirm:
            cur.execute("DELETE FROM invoice_line_items")
            conn.commit()
            st.success("✅ All entries deleted successfully.")
            st.rerun()
