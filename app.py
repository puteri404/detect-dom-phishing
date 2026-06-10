import streamlit as st
import pandas as pd
import joblib
import math
import jellyfish
import textdistance
from urllib.parse import urlparse

# ============================================================
# KONFIGURASI HALAMAN STREAMLIT
# ============================================================
st.set_page_config(page_title="Deteksi Typosquatting", page_icon="🛡️", layout="centered")

# ============================================================
# LOAD ARTEFAK MODEL & REFERENSI (CACHE AGAR CEPAT)
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('model_typosquatting_rf.pkl')
    tranco_refs = joblib.load('referensi_tranco.pkl')
    return model, tranco_refs

try:
    model, TRANCO_REFERENCE = load_artifacts()
except Exception as e:
    st.error(f"Error memuat file model. Pastikan file .pkl berada di folder yang sama! Error: {e}")
    st.stop()

# ============================================================
# KONFIGURASI FITUR (SAMA SEPERTI SAAT TRAINING)
# ============================================================
COMMON_TLDS = {"com", "org", "net", "edu", "gov", "id", "co", "io", "ac", "uk", "us", "au", "de", "fr", "jp"}
TLD_ABUSE_SCORE = {
    "tk": 0.95, "ml": 0.93, "ga": 0.92, "cf": 0.91, "gq": 0.90, "xyz": 0.80, "top": 0.78, "club": 0.70,
    "online": 0.68, "site": 0.65, "info": 0.55, "biz": 0.50, "live": 0.45, "net": 0.20, "org": 0.15, 
    "com": 0.10, "id": 0.05, "gov": 0.01, "edu": 0.01,
}
SUSPICIOUS_KEYWORDS = ["login", "secure", "update", "verify", "account", "banking", "payment", "confirm", "signin", "support", "service", "access", "portal", "wallet", "checkout"]
HOMOGLYPH_MAP = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "vv": "w", "rn": "m"}

# ============================================================
# FUNGSI EKSTRAKSI (DIAMBIL DARI SKRIP ANDA)
# ============================================================
def get_entropy(s: str) -> float:
    if not s: return 0.0
    freq = {c: s.count(c) / len(s) for c in set(s)}
    return -sum(p * math.log2(p) for p in freq.values())

def count_homoglyphs(domain: str) -> int:
    count, temp = 0, domain
    for fake in HOMOGLYPH_MAP:
        count += temp.count(fake)
        temp = temp.replace(fake, "")
    return count

def extract_features(domain: str) -> pd.DataFrame:
    domain = domain.strip().lower()
    if domain.startswith("http://") or domain.startswith("https://"):
        domain = urlparse(domain).netloc
    domain = domain.split("/")[0].split("?")[0].split("#")[0].split(":")[0]
    
    parts = domain.split(".")
    tld = parts[-1] if len(parts) >= 2 else ""
    sld = parts[-2] if len(parts) >= 2 else domain  
    subdomain_count = max(0, len(parts) - 2)

    # Menghitung Jarak String
    min_lev, min_dam_lev, max_jw, max_sim = float("inf"), float("inf"), 0.0, 0.0
    target_mirip = "Tidak ada"

    for brand in TRANCO_REFERENCE:
        if sld == brand: continue
            
        lev = jellyfish.levenshtein_distance(sld, brand)
        dam = textdistance.damerau_levenshtein(sld, brand)
        jw  = jellyfish.jaro_winkler_similarity(sld, brand)
        sim = 1 - (lev / max(len(sld), len(brand)))

        if lev < min_lev: min_lev = lev
        if dam < min_dam_lev: min_dam_lev = dam
        if sim > max_sim: max_sim = sim
        if jw > max_jw: 
            max_jw = jw
            target_mirip = brand

    # Menyusun fitur dalam bentuk DataFrame (agar sesuai dengan input Scikit-Learn)
    fitur_dict = {
        "domain_length": [len(domain)],
        "digit_count": [sum(c.isdigit() for c in sld)],
        "hyphen_count": [sld.count("-")],
        "entropy": [round(get_entropy(sld), 4)],
        "subdomain_count": [subdomain_count],
        "homoglyph_count": [count_homoglyphs(sld)],
        "levenshtein_distance": [min_lev],
        "damerau_levenshtein_distance": [min_dam_lev],
        "jaro_winkler_score": [round(max_jw, 4)],
        "max_similarity": [round(max_sim, 4)],
        "suspicious_keywords": [sum(kw in domain for kw in SUSPICIOUS_KEYWORDS)],
        "is_common_tld": [int(tld in COMMON_TLDS)],
        "tld_abuse_score": [TLD_ABUSE_SCORE.get(tld, 0.30)]
    }
    return pd.DataFrame(fitur_dict), target_mirip, min_lev, max_jw

# ============================================================
# ANTARMUKA PENGGUNA (UI)
# ============================================================
st.title("🛡️ Deteksi Phishing Typosquatting")
st.write("Sistem cerdas berbasis **Random Forest** untuk mendeteksi penipuan nama domain.")
st.markdown("---")

input_domain = st.text_input("Masukkan URL atau Nama Domain:", placeholder="contoh: klik-bca-update.com")

if st.button("Analisis Keamanan", type="primary"):
    if input_domain:
        with st.spinner("Mengekstrak fitur dan menganalisis metrik..."):
            # Ekstraksi Fitur
            df_fitur, target_mirip, lev, jw = extract_features(input_domain)
            
            # Prediksi dengan Model
            prediksi = model.predict(df_fitur)[0]
            probabilitas = model.predict_proba(df_fitur)[0]
            
            # =====================================================================
            # 4. KEAMANAN LAPIS KEDUA: RULE-BASED OVERRIDE (HYBRID SYSTEM)
            # =====================================================================
            alasan_override = ""
            
            # Jika ML bilang AMAN (0), tapi metrik jarak sangat mencurigakan:
            if prediksi == 0:
                # Kondisi 1: Beda 1-2 huruf DAN mengandung angka/homoglyph (contoh: g00gle.com)
                if (lev <= 2) and (jw >= 0.90) and (df_fitur['homoglyph_count'].values[0] > 0 or df_fitur['digit_count'].values[0] > 0):
                    prediksi = 1
                    probabilitas[1] = 0.95 # Paksa keyakinan jadi 95%
                    alasan_override = "Terdeteksi penggantian karakter visual (Homoglyph Typosquatting) pada entitas populer."
                
                # Kondisi 2: Ejaan sangat identik, beda 1 huruf saja (contoh: facebok.com)
                elif (lev == 1) and (jw >= 0.95):
                    prediksi = 1
                    probabilitas[1] = 0.92
                    alasan_override = "Terdeteksi kemiripan ejaan yang sangat identik (selisih 1 huruf) dengan entitas populer."
                    
        # ==============================================================================
        # 5. TAMPILKAN HASIL PREDIKSI
        # ==============================================================================
        st.markdown("---")
        st.subheader("Hasil Analisis:")
        
        if prediksi == 1:
            st.error(f"🚨 **PERINGATAN: Domain ini Terindikasi PHISHING / TYPOSQUATTING!**")
            st.metric(label="Tingkat Keyakinan Sistem", value=f"{probabilitas[1]*100:.2f}%")
            if alasan_override:
                st.warning(f"⚠️ **Intervensi Sistem Pakar:** {alasan_override}")
        else:
            st.success(f"✅ **AMAN: Domain ini Teridentifikasi NORMAL / SAH.**")
            st.metric(label="Tingkat Keyakinan Sistem", value=f"{probabilitas[0]*100:.2f}%")
