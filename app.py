# app.py
import streamlit as st
import requests
from datetime import datetime, timedelta
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="MandiBot — Your Mandi Price Assistant",
    page_icon="🌾",
    layout="wide"
)

# ----------------- Custom CSS -----------------
st.markdown("""
<style>
.main { background-color: #f1f8e9; }
.stButton > button { background-color: #2e7d32; color: white; border: none; border-radius: 5px;
    padding: 0.5rem 1rem; font-weight: bold; }
.stButton > button:hover { background-color: #1b5e20; }
.banner { background: linear-gradient(90deg, #2e7d32, #4caf50);
    color: white; padding: 1rem; border-radius: 10px;
    text-align: center; margin-bottom: 2rem; }
.footer { text-align: center; color: #2e7d32; font-style: italic;
    margin-top: 3rem; padding: 1rem; border-top: 2px solid #2e7d32; }
.price-result { background-color: #e8f5e8; border: 2px solid #2e7d32;
    border-radius: 10px; padding: 1rem; margin: 1rem 0;
    font-size: 1.2rem; font-weight: bold; color: #1b5e20; }
.error-message { background-color: #fff3e0; border: 2px solid #f57c00;
    border-radius: 10px; padding: 1rem; margin: 1rem 0; color: #e65100; }
</style>
""", unsafe_allow_html=True)

# ----------------- Banner -----------------
st.markdown("""
<div class="banner">
    <h1>🌾 MandiBot — Your Mandi Price Assistant</h1>
    <p>Real-time mandi prices for Farmers & Vendors</p>
</div>
""", unsafe_allow_html=True)

# ----------------- API Config -----------------
API_KEY = "579b464db66ec23bdd000001821e64e87e204b907bc5b548880a106d"
BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

# ----------------- Indian States & Districts -----------------
INDIAN_STATES_DISTRICTS = {
    "Maharashtra": ["Ahmednagar","Akola","Amravati","Aurangabad","Beed","Bhandara","Buldhana","Chandrapur","Dhule","Gadchiroli",
                     "Gondia","Hingoli","Jalgaon","Jalna","Kolhapur","Latur","Mumbai City","Mumbai Suburban","Nagpur","Nanded",
                     "Nandurbar","Nashik","Osmanabad","Palghar","Parbhani","Pune","Raigad","Ratnagiri","Sangli","Satara","Sindhudurg",
                     "Solapur","Thane","Wardha","Washim","Yavatmal"],
    # Add more states if required...
}

# ----------------- Embedding Model -----------------
# @st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# ----------------- Sample crops & semantic phrases -----------------
sample_crops = [
    "Tomato","Onion","Potato","Wheat","Rice","Maize","Cotton","Sugarcane","Soybean","Groundnut",
    "Chilli","Turmeric","Coriander","Cumin","Mustard","Sunflower","Jowar","Bajra","Tur","Gram",
    # Added common/region-specific crops to improve detection
    "Bhindi","Bitter gourd","Cucumber","Ginger","Banana","Brinjal","Cauliflower","Capsicum","Guar","Mint"
]

sample_phrases = [f"{c} price today" for c in sample_crops]

# @st.cache_resource
def get_phrase_embeddings(phrases):
    return embedding_model.encode(phrases, convert_to_tensor=True)

phrase_embeddings = get_phrase_embeddings(sample_phrases)

# ----------------- Helper Functions -----------------
# @st.cache_data(ttl=3600)
def fetch_mandi_data(commodity=None, date_filter=None):
    params = {"api-key": API_KEY, "format": "json", "limit": "500"}
    if commodity:
        params["filters[commodity]"] = commodity.title()
    if date_filter:
        params["filters[arrival_date]"] = date_filter
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("records", [])
    except:
        return []

def aggregate_prices(records):
    prices = []
    for r in records:
        try:
            val = float(str(r.get('modal_price','0')).replace(',', '').strip())
            prices.append(val)
        except:
            continue
    if not prices:
        return None, None, None
    return min(prices), max(prices), sum(prices)/len(prices)

def semantic_match(query, phrases, embeddings, model, threshold=60):
    query_emb = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_emb, embeddings)[0]
    best_idx = similarities.argmax().item()
    confidence = similarities[best_idx].item() * 100
    if confidence >= threshold:
        return phrases[best_idx], confidence
    return None, confidence

def parse_query(query, sample_crops, states_dict):
    query_lower = query.lower()
    commodity = None

    # 1) Exact substring/token match (highest priority)
    for c in sample_crops:
        if c.lower() in query_lower:
            commodity = c.title()
            break

    # 2) Fuzzy match fallback
    if not commodity:
        match, score, _ = process.extractOne(query, sample_crops, scorer=fuzz.WRatio)
        if score >= 70:  # tighten threshold to avoid accidental mismatches
            commodity = match.title()

    # 3) Semantic fallback as last resort
    if not commodity:
        sem_match, conf = semantic_match(query, sample_phrases, phrase_embeddings, embedding_model)
        if sem_match:
            commodity = sem_match.split()[0].title()
    locations = []

    for state, districts in states_dict.items():
        for district in districts:
            if district.lower() in query_lower:
                locations.append(district.title())
    if not locations:
        for state in states_dict.keys():
            if state.lower() in query_lower:
                locations.append(state.title())

    return commodity, locations

def filter_by_location(records, locations):
    if not locations:
        return records
    filtered = []
    for r in records:
        rec_state = r.get('state','').lower()
        rec_market = r.get('market','').lower()
        rec_district = r.get('district','').lower()
        for loc in locations:
            if loc.lower() in rec_state or loc.lower() in rec_market or loc.lower() in rec_district:
                filtered.append(r)
                break
    return filtered

def aggregate_prices(records):
    prices = []
    for r in records:
        try:
            val = float(str(r.get('modal_price','0')).replace(',', '').strip())
            prices.append(val)
        except:
            continue
    if not prices:
        return None, None, None
    return min(prices), max(prices), sum(prices)/len(prices)

def extract_last_context(history):
    last = {"commodity": None, "locations": [], "date_label": None, "date_filter": None}
    for msg in reversed(history):
        meta = msg.get("meta") or {}
        for k in ["commodity", "locations", "date_label", "date_filter"]:
            if last[k] in (None, [], "") and meta.get(k):
                last[k] = meta.get(k)
    return last

def resolve_context(query, history, sample_crops, states_dict):
    commodity, locations = parse_query(query, sample_crops, states_dict)
    ql = query.lower()
    if "yesterday" in ql:
        date_filter = (datetime.now() - timedelta(days=1)).strftime("%d/%m/%Y")
        date_label = "Yesterday"
    elif "today" in ql:
        date_filter = datetime.now().strftime("%d/%m/%Y")
        date_label = "Today"
    else:
        date_filter = None
        date_label = None

    if "history" not in st.session_state:
        st.session_state.history = []
    last = extract_last_context(st.session_state.history)

    if not commodity:
        commodity = last.get("commodity")
    if not locations:
        locations = last.get("locations") or []
    if not date_filter:
        date_filter = last.get("date_filter") or datetime.now().strftime("%d/%m/%Y")
        date_label = last.get("date_label") or "Today"
    if not date_label:
        date_label = "Today"

    return commodity, locations, date_filter, date_label

def get_records_with_fallback(commodity, locations, preferred_date, max_days_back=7):
    def to_label(days_back):
        if days_back == 0: return "Today"
        if days_back == 1: return "Yesterday"
        return (datetime.now() - timedelta(days=days_back)).strftime("%d %b %Y")

    for d in range(0, max_days_back + 1):
        if preferred_date:
            base = datetime.strptime(preferred_date, "%d/%m/%Y")
            dt = base - timedelta(days=d)
        else:
            dt = datetime.now() - timedelta(days=d)
        date_filter = dt.strftime("%d/%m/%Y")
        date_label = to_label(d) if preferred_date in (None, datetime.now().strftime("%d/%m/%Y")) else dt.strftime("%d %b %Y")
        recs = fetch_mandi_data(commodity, date_filter) if commodity else fetch_mandi_data(date_filter=date_filter)
        filtered = filter_by_location(recs, locations)
        if filtered:
            return filtered, date_filter, date_label
    return [], preferred_date, ("Today" if preferred_date == datetime.now().strftime("%d/%m/%Y") else "Selected date")

def format_price_response(commodity, date_label, records_filtered):
    min_price, max_price, avg_price = aggregate_prices(records_filtered)
    markets_list = sorted(set([r.get('market','Unknown') for r in records_filtered]))[:5]
    states_list = sorted(set([r.get('state','Unknown') for r in records_filtered]))[:5]

    # Build vertical markdown with markets/states each on their own line
    markets_md = "\n".join([f"  • {m}" for m in markets_list]) if markets_list else "  • Unknown"
    states_md = "\n".join([f"  • {s}" for s in states_list]) if states_list else "  • Unknown"

    txt = (
        f"🌾 **{commodity.title()} — {date_label}**\n\n"
        f"**Price range:**\n  ₹{min_price:.0f} – ₹{max_price:.0f}/quintal\n\n"
        f"**Average price:**\n  ₹{avg_price:.0f}/quintal\n\n"
        f"**Top markets:**\n{markets_md}\n\n"
        f"**States:**\n{states_md}\n\n"
        f"Tip: Prices vary by market and quality. 📈"
    )
    return txt

# ----------------- Main App -----------------
def main():
    # Removed role selection to simplify UX
    st.markdown("---")
    st.subheader("Chat with MandiBot")

    # Sidebar quick filters (date & location) to make queries faster
    with st.sidebar:
        st.markdown("### Quick Filters")
        # Use a date object as the default for the date_input (st.date_input returns a date)
        sidebar_date = st.date_input("Select date", datetime.now().date())
        sidebar_location = st.text_input("Location (city/district/state)", "")
        st.markdown("---")
        st.markdown("Need help? Try queries like: \n• Rate of onion in Pune today\n• Bhindi price in Ahmednagar yesterday")

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    if not st.session_state.history:
        greeting = {
            "role": "assistant",
            "content": "Namaste! 🌾 I’m MandiBot. Ask me mandi prices like ‘Rate of onion in Ahmednagar today’, ‘What about wheat in Pune yesterday?’, or ‘Show me crops available in Nashik’.",
            "meta": {}
        }
        st.session_state.history.append(greeting)

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input(placeholder="Type your question… e.g., Rate of onion in Ahmednagar today")
    if user_query:
        st.session_state.history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        commodity, locations, date_filter, date_label = resolve_context(user_query, st.session_state.history, sample_crops, INDIAN_STATES_DISTRICTS)

        # Apply sidebar overrides if user didn't include them in the query
        if sidebar_location and not locations:
            locations = [sidebar_location.title()]
        # Convert sidebar_date to expected format if user didn't specify date in query
        if sidebar_date and (not date_filter or date_label == "Today"):
            # Use the sidebar date as authoritative when set by the user
            date_filter = sidebar_date.strftime("%d/%m/%Y")
            today_date = datetime.now().date()
            if sidebar_date == today_date:
                date_label = "Today"
            elif sidebar_date == (today_date - timedelta(days=1)):
                date_label = "Yesterday"
            else:
                date_label = sidebar_date.strftime("%d %b %Y")

        # Top 3 semantic suggestions if commodity unknown
        top_suggestions = []
        if not commodity:
            query_emb = embedding_model.encode(user_query, convert_to_tensor=True)
            similarities = util.cos_sim(query_emb, phrase_embeddings)[0]
            top_idxs = similarities.topk(3).indices.tolist()
            top_suggestions = [sample_phrases[i] for i in top_idxs]

        records_filtered = []
        used_date_label = None
        if commodity:
            spinner_text = f"Fetching {commodity.title()} prices for {', '.join(locations) if locations else 'selected location'}..."
            with st.spinner(spinner_text):
                records_filtered, used_date_filter, used_date_label = get_records_with_fallback(commodity, locations, date_filter)

        if commodity and records_filtered:
            reply = format_price_response(commodity, used_date_label or date_label, records_filtered)
            meta = {"commodity": commodity, "locations": locations, "date_label": used_date_label or date_label, "date_filter": used_date_filter if 'used_date_filter' in locals() else date_filter}
        elif top_suggestions:
            reply = "⚠️ I couldn't clearly understand the crop. Did you mean one of these?\n\n" + "\n".join([f"• {s}" for s in top_suggestions])
            meta = {"commodity": None, "locations": locations, "date_label": date_label, "date_filter": date_filter}
        else:
            # Spinner for fallback search across all commodities
            spinner_text = f"Searching available crops in {', '.join(locations) if locations else 'your area'}..."
            with st.spinner(spinner_text):
                all_filtered, used_date_filter2, used_date_label2 = get_records_with_fallback(None, locations, date_filter)
            if all_filtered:
                displayed = []
                seen = set()
                for r in all_filtered:
                    c = r.get('commodity','Unknown')
                    if c and c not in seen:
                        displayed.append(f"• {c} ({r.get('market','Unknown')}, {r.get('state','Unknown')})")
                        seen.add(c)
                    if len(displayed) >= 12:
                        break
                loc_label = ", ".join(locations) if locations else "your area"
                reply = f"⚠️ I couldn’t find reliable data for that crop in {loc_label}.\n\nHere are other crops available:\n\n" + "\n".join(displayed)
                meta = {"commodity": None, "locations": locations, "date_label": used_date_label2 or date_label, "date_filter": used_date_filter2 or date_filter}
            else:
                loc_label = ", ".join(locations) if locations else "your area"
                reply = f"😞 No mandi data found for {loc_label}."
                meta = {"commodity": None, "locations": locations, "date_label": date_label, "date_filter": date_filter}

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.history.append({"role": "assistant", "content": reply, "meta": meta})

        # Static tip section (no role-based UI)
        st.info("💡 Tip: Compare prices across multiple markets for best rates and monitor trends for informed decisions.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        Made with ❤️ for India's Agriculture
        <br>
        <small>Developed by 🤍 Shrikant Pawar</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
