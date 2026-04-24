import streamlit as st
import pandas as pd
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from PIL import Image

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MEDICINE_PRICES = {
    "metronidazole": ("Metronidazole 250mg", 45),
    "amoxicillin": ("Amoxicillin 500mg", 85),
    "doxycycline": ("Doxycycline 100mg", 120),
    "enrofloxacin": ("Enrofloxacin 50mg", 150),
    "prednisolone": ("Prednisolone 5mg", 60),
    "ivermectin": ("Ivermectin 1%", 95),
    "fenbendazole": ("Fenbendazole 150mg", 110),
    "ondansetron": ("Ondansetron 4mg", 75),
    "cerenia": ("Cerenia (Maropitant) 16mg", 580),
    "maropitant": ("Cerenia (Maropitant) 16mg", 580),
    "clavulanic": ("Amoxicillin-Clavulanate 625mg", 220),
    "ampicillin": ("Ampicillin 500mg", 95),
    "ketoprofen": ("Ketoprofen 20mg", 130),
    "meloxicam": ("Meloxicam 1.5mg/ml", 180),
    "furosemide": ("Furosemide 40mg", 55),
    "atenolol": ("Atenolol 25mg", 65),
    "enalapril": ("Enalapril 5mg", 90),
    "ketoconazole": ("Ketoconazole 200mg", 160),
    "itraconazole": ("Itraconazole 100mg", 310),
    "fluconazole": ("Fluconazole 150mg", 140),
    "chlorhexidine": ("Chlorhexidine Wash 4%", 200),
    "praziquantel": ("Praziquantel 600mg", 175),
    "albendazole": ("Albendazole 400mg", 85),
    "vitamin": ("Vitamin B Complex Injection", 65),
    "ringer": ("Ringer's Lactate 500ml IV", 95),
    "fluids": ("IV Normal Saline 500ml", 90),
    "antiserum": ("Parvovirus Antiserum 5ml", 850),
    "dewormer": ("Broad Spectrum Dewormer", 100),
    "antihistamine": ("Chlorpheniramine 4mg", 40),
    "diphenhydramine": ("Diphenhydramine 25mg", 55),
    "cefovecin": ("Cefovecin Sodium 80mg/ml", 420),
    "azithromycin": ("Azithromycin 250mg", 145),
    "tylosin": ("Tylosin 100mg", 190),
    "default": ("Prescribed Medication", 200),
}

MEDICINE_IMAGES = {
    "injection": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Medicine_Tablets.jpg/320px-Medicine_Tablets.jpg",
    "tablet": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Colorful_pills.jpg/320px-Colorful_pills.jpg",
    "syrup": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Vials.jpg/320px-Vials.jpg",
    "default": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Veterinary_pharmaceutical.jpg/320px-Veterinary_pharmaceutical.jpg",
}

NEGATION_WORDS = {
    'not', 'no', 'never', 'without', 'lack', 'lacking', 'absent',
    'neither', 'nor', 'cannot', "can't", "won't", "doesn't",
    "isn't", "hasn't", "haven't", 'barely', 'hardly', 'rarely', 'unable'
}

URGENCY_KEYWORDS = {
    'high': [
        'collapse', 'seizure', 'blood', 'bloody', 'convulsion', 'unconscious',
        'paralysis', 'breathing', 'choking', 'pale gums', 'rapid heart',
        'extreme weakness', 'unresponsive', 'shock'
    ],
    'medium': [
        'vomiting', 'diarrhea', 'lethargy', 'fever', 'dehydration',
        'loss of appetite', 'limping', 'coughing', 'sneezing', 'discharge'
    ],
}

# ─────────────────────────────────────────────
#  DISEASE ADVICE TEMPLATES (keyed by category)
# ─────────────────────────────────────────────
CATEGORY_ADVICE = {
    "Viral": {
        "overview": (
            "Viral infections in pets can range from moderately uncomfortable to life-threatening, "
            "and they require prompt veterinary attention because there is no direct cure for viruses — "
            "treatment focuses on supportive care to help your pet's immune system fight the infection. "
            "Based on the symptoms you described — {symptoms} — your {animal} may be experiencing "
            "a {category} illness called {disease}, which commonly affects {animal}s and can progress "
            "quickly, especially in young, elderly, or unvaccinated animals. It is important not to "
            "underestimate this condition even if your pet seems only mildly unwell right now."
        ),
        "home_care": (
            "Before you are able to see a vet, keep your {animal} warm, calm, and as comfortable as "
            "possible. Offer small sips of fresh water frequently to prevent dehydration — even if your "
            "pet refuses to drink, gentle encouragement every 15 to 20 minutes can help. Do not force-feed "
            "solid food if your pet is vomiting or showing no interest; instead, a bland diet of boiled "
            "chicken and plain rice in very small amounts can be introduced once vomiting has subsided for "
            "a few hours. Keep your {animal} away from other pets to prevent potential spread, and isolate "
            "their food bowls and bedding to limit contamination."
        ),
        "emergency_signs": (
            "You must take your pet to an emergency vet immediately if you notice any of the following: "
            "complete unresponsiveness or collapse, bloody vomit or stool, extreme difficulty breathing, "
            "pale or bluish gums, seizures, or if your {animal} has not had any water in over 12 hours. "
            "These signs indicate the condition may be progressing into a dangerous stage that requires "
            "IV fluids, antiviral support, or intensive care that cannot be managed at home. Do not wait "
            "to see if things improve on their own if any of these signs appear."
        ),
        "prognosis": (
            "With timely veterinary intervention, many pets recover well from viral infections, especially "
            "when treatment begins early. Vaccinated pets generally have a much stronger immune response, "
            "so this is also a good reminder to keep your {animal}'s vaccination schedule up to date going "
            "forward. Stay close to your pet, monitor their condition every few hours, and trust your "
            "instincts — you know your animal best, and your prompt attention has already made a difference."
        ),
    },
    "Bacterial": {
        "overview": (
            "Bacterial infections are among the more common reasons pets fall ill, and the good news is "
            "they are generally very treatable when caught in time. Based on the symptoms you've described "
            "— {symptoms} — your {animal} may be dealing with a {category} condition known as {disease}. "
            "Bacterial illnesses can affect different organ systems depending on the type, and symptoms "
            "can sometimes appear suddenly or worsen quickly if left unaddressed. The bacteria involved "
            "can sometimes be opportunistic, meaning they take hold when the immune system is already "
            "under stress from another cause such as a recent illness, injury, or environmental stressor."
        ),
        "home_care": (
            "Until you can reach a veterinarian, ensure your {animal} has a quiet, clean resting area "
            "away from drafts and other pets. Keep any affected skin or wound areas clean and dry if "
            "relevant. Offer fresh water consistently and avoid switching to any new foods, as digestive "
            "stress can worsen the infection's impact. Never administer antibiotics meant for humans — "
            "many human antibiotics are toxic to pets, and even pet-safe antibiotics require the correct "
            "dosage prescribed by a vet based on your pet's weight and the specific bacteria involved."
        ),
        "emergency_signs": (
            "Seek emergency care right away if your {animal} develops a high fever (normal is 38–39.2°C "
            "for both dogs and cats), shows severe swelling, develops an abscess that bursts or oozes, "
            "loses consciousness, or if the symptoms worsen rapidly over a few hours. Septicemia — a "
            "bacterial infection spreading into the bloodstream — is a life-threatening complication that "
            "can develop from untreated bacterial illness, and it requires immediate hospitalisation."
        ),
        "prognosis": (
            "Bacterial infections, when treated with the correct antibiotics and supportive care, have an "
            "excellent prognosis in most cases. Your vet will likely prescribe a full course of antibiotics "
            "— it is crucial to complete the entire course even if your {animal} appears to recover quickly, "
            "to prevent antibiotic resistance and relapse. Most pets show noticeable improvement within "
            "48 to 72 hours of starting treatment, and a full recovery is very achievable."
        ),
    },
    "Parasitic": {
        "overview": (
            "Parasitic conditions are extremely common in both dogs and cats, and they can range from "
            "mildly irritating to seriously harmful depending on the type and severity of the infestation. "
            "Based on what you've described — {symptoms} — your {animal} may be suffering from {disease}, "
            "a {category} condition that can affect the skin, digestive system, or internal organs "
            "depending on the parasite involved. Some parasites also pose a zoonotic risk, meaning they "
            "can potentially be transmitted to humans in the household, so hygiene precautions are "
            "especially important while your pet is being diagnosed and treated."
        ),
        "home_care": (
            "While waiting for your vet appointment, inspect your {animal}'s coat thoroughly under good "
            "lighting for fleas, ticks, or mites — pay close attention to behind the ears, around the "
            "neck, and under the legs. Wash your pet's bedding and soft toys in hot water and vacuum your "
            "home thoroughly, especially carpets and upholstered furniture. Avoid applying any "
            "over-the-counter anti-parasite treatments without vet guidance, as some products contain "
            "permethrin which is highly toxic to cats. Keep your {animal} away from outdoor soil or "
            "grass if internal parasites are suspected."
        ),
        "emergency_signs": (
            "Rush to the vet if your {animal} shows pale gums (indicating possible anemia from "
            "blood-sucking parasites), extreme weight loss over a short period, bloody diarrhea, severe "
            "skin wounds from scratching, or signs of neurological issues such as wobbling or seizures, "
            "which can occur with some tick-borne diseases. Heartworm in particular can cause "
            "life-threatening cardiac complications and requires urgent, specialised treatment that "
            "should not be delayed."
        ),
        "prognosis": (
            "Most parasitic conditions are very manageable with the right medication and environmental "
            "decontamination. Your vet will recommend a dewormer, anti-parasite spot-on, or other "
            "targeted treatment depending on the parasite identified. Reinfection is common, so your "
            "vet will likely suggest a regular prevention schedule going forward — monthly preventatives "
            "are one of the best long-term investments you can make in your {animal}'s ongoing health."
        ),
    },
    "Fungal": {
        "overview": (
            "Fungal infections, while less talked about than bacterial or viral ones, can cause "
            "significant discomfort and in some cases serious health complications in pets. "
            "Your {animal} may be experiencing {disease}, a {category} condition that commonly "
            "presents with symptoms such as {symptoms}. Fungi thrive in warm, moist environments "
            "and can affect the skin, nails, respiratory system, or internal organs depending on the "
            "species involved. Some fungal infections like ringworm are also contagious to humans, so "
            "handling your pet and maintaining strict hygiene is particularly important right now."
        ),
        "home_care": (
            "Keep affected areas dry and clean — moisture encourages fungal growth, so avoid unnecessary "
            "baths unless specifically recommended by your vet. If bathing is needed, dry your {animal} "
            "thoroughly and promptly afterward using a clean towel. Do not share towels or grooming tools "
            "between pets or with humans in the household. Wear gloves when handling your {animal}'s "
            "affected skin areas and wash your hands thoroughly after every contact. Where possible, "
            "improve ventilation in your home and reduce humidity, and wash your pet's bedding frequently "
            "in hot water to kill spores."
        ),
        "emergency_signs": (
            "See a vet urgently if the affected skin areas spread rapidly, develop open sores or bleeding, "
            "your {animal} develops respiratory distress (which can happen with systemic fungal infections), "
            "shows signs of neurological changes such as head tilting or stumbling, or if multiple pets in "
            "the household begin showing similar symptoms simultaneously. Systemic fungal infections that "
            "reach the lungs or brain are medical emergencies requiring aggressive antifungal therapy and "
            "close monitoring."
        ),
        "prognosis": (
            "Superficial fungal infections like ringworm typically respond well to topical and oral "
            "antifungal medications over several weeks. Systemic infections require longer treatment "
            "courses and more regular monitoring. Consistency with medication is key — stopping treatment "
            "too early is the most common reason infections return. With patience and the correct treatment "
            "plan from your vet, most pets make a full and lasting recovery."
        ),
    },
    "default": {
        "overview": (
            "Based on the symptoms you've described — {symptoms} — your {animal} may be experiencing "
            "a condition called {disease}. While it's important not to jump to conclusions without a "
            "formal veterinary examination, understanding what might be happening can help you take "
            "the right steps to support your pet's health and comfort right now. Every animal is "
            "different, and symptoms can sometimes overlap between conditions, which is exactly why "
            "a thorough professional examination and possibly some diagnostic tests are so valuable "
            "in getting to the right answer quickly."
        ),
        "home_care": (
            "Right now, your most important job is to keep your {animal} calm, comfortable, and "
            "well-rested. Offer fresh water at all times and keep food light and easy to digest. "
            "Avoid any strenuous exercise or play until you have a better understanding of what's "
            "going on. Monitor your pet closely — take note of any new symptoms, changes in bathroom "
            "habits, eating patterns, or energy levels, as this information will be extremely valuable "
            "when you speak with your vet and will help them make a faster, more accurate assessment."
        ),
        "emergency_signs": (
            "Do not delay veterinary care if your {animal} stops breathing normally, loses consciousness, "
            "develops sudden swelling of the face or throat, vomits or passes blood, shows signs of "
            "extreme pain such as crying out, guarding a body part, or inability to stand, or if any "
            "current symptoms worsen significantly within a short window of time. Trust your instincts "
            "— as a pet owner, you are often the very first to notice when something is seriously wrong, "
            "and acting on that feeling early can make all the difference."
        ),
        "prognosis": (
            "Without a confirmed diagnosis it is difficult to give a specific prognosis, but prompt "
            "veterinary attention dramatically improves outcomes for the vast majority of pet illnesses. "
            "The fact that you noticed these symptoms and are actively seeking guidance already puts your "
            "{animal} in a much better position than if the signs had gone unnoticed. Take a deep breath "
            "— you're doing the right thing by paying attention and taking action today."
        ),
    }
}


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("pet_disease_nlp_dataset_500.csv")
    return df


# ─────────────────────────────────────────────
#  NLP: NEGATION-AWARE KEYWORD EXTRACTION
# ─────────────────────────────────────────────
def extract_keywords_with_negation(text):
    """
    Clause-level negation-aware tokenizer.
    Splits text on punctuation into clauses, then within each clause
    detects negation trigger words and tags all following tokens as negated.
    Example: 'not eating' → negated('eating'), 'no appetite' → negated('appetite')
    Positive tokens before the negation word are preserved correctly.
    """
    text_clean = text.lower()
    phrases = re.split(r'[,;.\n!?]', text_clean)

    positive_symptoms = []
    negated_symptoms = []
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have',
        'my', 'their', 'its', 'and', 'or', 'but', 'with', 'for',
        'in', 'on', 'at', 'to', 'of', 'from', 'by', 'be', 'been',
        'him', 'her', 'she', 'he', 'it', 'his', 'our', 'your', 'since',
        'when', 'while', 'also', 'very', 'quite', 'still', 'just', 'now'
    }

    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue

        words = phrase.split()
        negated = False
        symptom_words = []

        for w in words:
            clean = re.sub(r"[^a-z']", '', w)
            if not clean:
                continue
            if clean in NEGATION_WORDS:
                # flush accumulated words before negation as positive
                if symptom_words and not negated:
                    positive_symptoms.append(' '.join(symptom_words))
                symptom_words = []
                negated = True
                continue
            if clean in stop_words or len(clean) <= 2:
                continue
            symptom_words.append(clean)

        if symptom_words:
            symptom = ' '.join(symptom_words)
            if negated:
                negated_symptoms.append(symptom)
            else:
                positive_symptoms.append(symptom)

    return positive_symptoms, negated_symptoms


# ─────────────────────────────────────────────
#  NLP: TF-IDF SYMPTOM MATCHING (sklearn)
# ─────────────────────────────────────────────
@st.cache_resource
def build_tfidf_model(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, analyzer='word')
    corpus = df['symptoms'].str.lower().tolist()
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def find_top_diseases(query_symptoms, df, vectorizer, matrix, animal_type, top_n=3):
    """TF-IDF cosine similarity matching, filtered by animal type."""
    animal_df = df[df['animal_type'].str.lower() == animal_type.lower()].copy()
    animal_df = animal_df.reset_index(drop=True)
    animal_matrix = vectorizer.transform(animal_df['symptoms'].str.lower().tolist())

    query = ' '.join(query_symptoms)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, animal_matrix).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'disease': animal_df.iloc[idx]['disease_name'],
            'category': animal_df.iloc[idx]['disease_category'],
            'medicines': animal_df.iloc[idx]['recommended_medicines'],
            'score': round(float(similarities[idx]), 3),
            'matched_symptoms': animal_df.iloc[idx]['symptoms']
        })
    return results


# ─────────────────────────────────────────────
#  RULE-BASED ADVICE GENERATOR  (zero API calls)
# ─────────────────────────────────────────────
def detect_urgency(symptoms_list):
    joined = ' '.join(symptoms_list).lower()
    for kw in URGENCY_KEYWORDS['high']:
        if kw in joined:
            return 'high'
    for kw in URGENCY_KEYWORDS['medium']:
        if kw in joined:
            return 'medium'
    return 'low'


def generate_advice(animal, disease, category, symptoms, negated_symptoms):
    """
    Builds a rich, multi-paragraph advisory from category-keyed templates.
    Fully offline — no external API required.
    """
    template = CATEGORY_ADVICE.get(category, CATEGORY_ADVICE["default"])
    symptoms_str = ', '.join(symptoms[:6]) if symptoms else 'various symptoms'
    fmt = dict(animal=animal, disease=disease, category=category, symptoms=symptoms_str)

    para1 = template["overview"].format(**fmt)
    para2 = template["home_care"].format(**fmt)
    para3 = template["emergency_signs"].format(**fmt)
    para4 = template["prognosis"].format(**fmt)

    # Weave in negated symptoms naturally into the overview paragraph
    if negated_symptoms:
        neg_str = ', '.join(negated_symptoms)
        para1 += (
            f" It is also worth noting that your {animal} does NOT currently appear to be showing "
            f"{neg_str} — the absence of these symptoms is useful diagnostic context that your vet "
            f"will want to hear about when you describe the situation."
        )

    urgency = detect_urgency(symptoms)
    urgency_label = {
        'high': '🔴 HIGH — seek emergency vet care immediately',
        'medium': '🟡 MODERATE — schedule a vet visit today',
        'low': '🟢 LOWER — monitor closely, vet visit within 24–48 hrs'
    }[urgency]

    return urgency_label, para1, para2, para3, para4


# ─────────────────────────────────────────────
#  MEDICINE PARSER
# ─────────────────────────────────────────────
def parse_medicines(medicines_str):
    raw = [m.strip() for m in medicines_str.split(',') if m.strip()]
    parsed = []
    for med in raw:
        med_lower = med.lower()
        price_info = None
        for key, (name, price) in MEDICINE_PRICES.items():
            if key in med_lower:
                price_info = (name, price)
                break
        if not price_info:
            price_info = (med, MEDICINE_PRICES["default"][1])

        if any(x in med_lower for x in ['injection', 'iv', 'antiserum', 'serum']):
            img_key = "injection"
        elif any(x in med_lower for x in ['syrup', 'solution', 'wash', 'liquid']):
            img_key = "syrup"
        else:
            img_key = "tablet"

        parsed.append({
            'name': med,
            'display_name': price_info[0],
            'price': price_info[1],
            'image_key': img_key
        })
    return parsed


# ─────────────────────────────────────────────
#  UI: MEDICINE CARD
# ─────────────────────────────────────────────
def render_medicine_card(med):
    img_url = MEDICINE_IMAGES.get(med['image_key'], MEDICINE_IMAGES['default'])
    st.markdown(f"""
    <div style="border:1px solid #e2e8f0; border-radius:12px; padding:14px;
                background:#f8fafc; margin-bottom:10px;
                box-shadow:0 1px 3px rgba(0,0,0,0.07);">
        <div style="font-weight:700; font-size:15px; color:#1e293b; margin-bottom:6px;">
            💊 {med['name']}
        </div>
        <div style="font-size:13px; color:#64748b; margin-bottom:8px;">
            Generic: {med['display_name']}
        </div>
        <div style="display:inline-block; background:#dcfce7; color:#166534;
                    padding:4px 12px; border-radius:999px; font-size:13px; font-weight:600;">
            ₹{med['price']} approx.
        </div>
    </div>
    """, unsafe_allow_html=True)
    try:
        r = requests.get(img_url, timeout=4)
        if r.status_code == 200:
            img = Image.open(BytesIO(r.content))
            st.image(img, width=160, caption=med['display_name'])
    except Exception:
        st.markdown("🖼️ *Image unavailable*")


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="🐾 Pet Disease Advisor",
        page_icon="🐾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-title { font-size:2.4rem; font-weight:800; color:#1e293b; margin-bottom:0.2rem; }
    .subtitle   { font-size:1rem; color:#64748b; margin-bottom:1.5rem; }
    .section-header {
        font-size:1.3rem; font-weight:700; color:#0f172a;
        margin:1.5rem 0 0.7rem 0; padding-bottom:6px;
        border-bottom:2px solid #e2e8f0;
    }
    .symptom-chip {
        display:inline-block; background:#dbeafe; color:#1e40af;
        padding:4px 12px; border-radius:999px; margin:3px;
        font-size:13px; font-weight:500;
    }
    .negated-chip {
        display:inline-block; background:#fee2e2; color:#991b1b;
        padding:4px 12px; border-radius:999px; margin:3px;
        font-size:13px; font-weight:500; text-decoration:line-through;
    }
    .advice-para {
        background:#f8fafc; border-left:4px solid #6366f1;
        border-radius:0 10px 10px 0; padding:16px 20px;
        line-height:1.85; color:#1a1a1a; margin-bottom:12px; font-size:15px;
    }
    .advice-green  { border-left-color:#22c55e !important; background:#f0fdf4 !important; }
    .advice-red    { border-left-color:#ef4444 !important; background:#fef2f2 !important; }
    .advice-purple { border-left-color:#a855f7 !important; background:#faf5ff !important; }
    .model-badge {
        background:#0f766e; color:white; padding:4px 14px;
        border-radius:999px; font-size:12px; font-weight:600;
        display:inline-block; margin-bottom:12px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────
    with st.sidebar:
        st.markdown("## 🐾 Pet Advisor")
        st.markdown("---")
        st.markdown("""
**How it works:**
1. Select your animal
2. Paste symptom description
3. Get instant diagnosis + advice

---
**🧠 NLP Pipeline:**
- **Keyword Extraction:** Clause-level negation-aware tokenizer
- **Disease Matching:** TF-IDF + Cosine Similarity (sklearn)
- **Advice Generation:** Rule-based template engine

*100% offline — no API key required*

---
⚠️ *Advisory only. Always consult a licensed veterinarian.*
        """)
        st.markdown("---")
        st.markdown("**Dataset:** 500 pet disease records (dog & cat)")

    # ── Header ───────────────────────────────────
    st.markdown('<div class="main-title">🐾 Pet Disease Advisor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Describe your pet\'s symptoms in plain language — '
        'keywords are extracted (with negation handling), diseases matched, '
        'and care advice generated instantly.</div>',
        unsafe_allow_html=True
    )

    # ── Inputs ───────────────────────────────────
    col1, col2 = st.columns([1, 2])
    with col1:
        animal = st.selectbox("🐶🐱 Select your animal", ["Dog", "Cat"])
    with col2:
        st.markdown("**📝 Describe symptoms (natural language)**")
        symptom_text = st.text_area(
            label="symptom_input",
            label_visibility="collapsed",
            placeholder=(
                "e.g. My dog has been vomiting since yesterday, not eating at all, "
                "severe lethargy, not moving much, bloody stool, signs of dehydration..."
            ),
            height=140
        )

    analyze_btn = st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True)

    # ── Analysis ─────────────────────────────────
    if analyze_btn and symptom_text.strip():
        df = load_data()
        vectorizer, matrix = build_tfidf_model(df)

        with st.spinner("🔬 Extracting symptoms with negation detection..."):
            positive_symptoms, negated_symptoms = extract_keywords_with_negation(symptom_text)

        with st.spinner("📊 Matching against disease database..."):
            if not positive_symptoms:
                st.warning("⚠️ No positive symptoms detected. Please provide more descriptive symptoms.")
                return
            results = find_top_diseases(positive_symptoms, df, vectorizer, matrix, animal, top_n=3)

        # ── Keyword Chips ─────────────────────────
        st.markdown('<div class="section-header">🏷️ Extracted Keywords</div>', unsafe_allow_html=True)
        chips_html = "".join(
            [f'<span class="symptom-chip">✓ {s}</span>' for s in positive_symptoms] +
            [f'<span class="negated-chip">✗ {s}</span>' for s in negated_symptoms]
        )
        st.markdown(chips_html, unsafe_allow_html=True)
        if negated_symptoms:
            st.caption("🔴 Red/strikethrough = negated symptoms (e.g. 'not eating') — excluded from disease matching, included in advice")

        # ── Disease Cards ─────────────────────────
        st.markdown('<div class="section-header">🔬 Top Matched Conditions</div>', unsafe_allow_html=True)
        best = results[0]

        cols = st.columns(3)
        for i, (res, col) in enumerate(zip(results, cols)):
            with col:
                border = "#6366f1" if i == 0 else "#cbd5e1"
                label = "🥇 Most Likely" if i == 0 else f"#{i+1} Possible"
                st.markdown(f"""
                <div style="border:2px solid {border}; border-radius:12px; padding:14px; background:#fafafa;">
                    <div style="font-size:11px; font-weight:700; color:{border}; margin-bottom:4px;">{label}</div>
                    <div style="font-size:17px; font-weight:800; color:#0f172a;">{res['disease']}</div>
                    <div style="font-size:12px; color:#64748b; margin:4px 0;">{res['category']}</div>
                    <div style="background:#e0e7ff; color:#3730a3; padding:3px 10px;
                                border-radius:999px; font-size:12px; display:inline-block;">
                        Similarity: {res['score']:.0%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Advice ────────────────────────────────
        st.markdown('<div class="section-header">🩺 Primary Care Advice</div>', unsafe_allow_html=True)
        st.markdown('<div class="model-badge">⚙️ Rule-based NLP Engine — Fully Offline, No API Key</div>', unsafe_allow_html=True)

        urgency_label, p_overview, p_homecare, p_emergency, p_prognosis = generate_advice(
            animal=animal.lower(),
            disease=best['disease'],
            category=best['category'],
            symptoms=positive_symptoms,
            negated_symptoms=negated_symptoms
        )

        urgency_color = (
            "#dc2626" if "HIGH" in urgency_label else
            "#d97706" if "MODERATE" in urgency_label else
            "#16a34a"
        )
        st.markdown(f"""
        <div style="background:{urgency_color}18; border:1.5px solid {urgency_color};
                    border-radius:10px; padding:10px 18px; font-weight:700;
                    font-size:14px; color:{urgency_color}; margin-bottom:14px;">
            ⚡ Urgency Level: {urgency_label}
        </div>
        """, unsafe_allow_html=True)

        for title, body, css in [
            ("📋 What's Happening",              p_overview,  ""),
            ("🏠 Immediate Home Care",            p_homecare,  "advice-green"),
            ("🚨 When to Go to Emergency Vet",    p_emergency, "advice-red"),
            ("💚 Prognosis & Recovery",           p_prognosis, "advice-purple"),
        ]:
            st.markdown(f"**{title}**")
            st.markdown(f'<div class="advice-para {css}">{body}</div>', unsafe_allow_html=True)

        # ── Medicines ─────────────────────────────
        st.markdown('<div class="section-header">💊 Recommended Medicines & Pricing</div>', unsafe_allow_html=True)
        st.caption(f"Based on diagnosis: **{best['disease']}** — Approximate Indian market rates (₹)")

        medicines = parse_medicines(best['medicines'])
        total_cost = sum(m['price'] for m in medicines)

        med_cols = st.columns(min(len(medicines), 3))
        for i, med in enumerate(medicines):
            with med_cols[i % 3]:
                render_medicine_card(med)

        st.markdown(f"""
        <div style="background:#fef9c3; border:1px solid #fde047; border-radius:10px;
                    padding:14px 20px; margin-top:10px;">
            <strong>💰 Estimated Total:</strong> ₹{total_cost} – ₹{total_cost + 200}
            <span style="font-size:12px; color:#78716c;"> (varies by region & vet)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.error(
            "⚠️ **Medical Disclaimer:** This tool provides informational guidance only and is NOT a "
            "substitute for professional veterinary diagnosis. Always consult a licensed veterinarian "
            "before administering any medication to your pet."
        )

    elif analyze_btn:
        st.warning("Please enter your pet's symptoms above before analyzing.")


if __name__ == "__main__":
    main()