# -*- coding: utf-8 -*-
"""
LP Modeli: Linear Programming for Aggregate Production Planning in a Textile Company
Makale referansı: "Linear Programming for Aggregate Production Planning in a Textile Company"
DOI: 10.5604/01.3001.0012.2525

Bu kod, makaledeki veri setleri (Table 1-6) esas alınarak; tüm parametre, veri seti, karar değişkeni, 
amac fonksiyonu ve kısıtlar TL cinsinden (38 TL/USD kuru uygulanarak) uygulanmıştır. 
Ayrıca, Streamlit ve Plotly kullanılarak interaktif, modern ve profesyonel görselleştirmeler sunulmaktadır.
"""

import pulp
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import locale

# ---------------------------
# Locale Ayarları: ABD sayısal formatı (binlik ayracı nokta, ondalık ayracı virgül)
# ---------------------------
try:
    locale.setlocale(locale.LC_NUMERIC, 'en_US.UTF-8')  # Varsayılan
except locale.Error:
    locale.setlocale(locale.LC_NUMERIC, 'C')  # Alternatif

# ---------------------------
# 1. VERİ SETLERİ VE PARAMETRELER (TL cinsinden)
# ---------------------------

# İndeksler:
I = 5   # Süreçler: 1=Warping, 2=Warping Check, 3=Gumming, 4=Weaving, 5=Weaving Check
T = 12  # Aylar: 1...12
L = 3   # Ürün Hatları: Line 1, Line 2, Line 3

# --- Tablo 5: Çalışan maliyetleri (USD→TL: 38 TL/USD) ---
CE_TL = {
    1: 696.7 * 38,   # Warping
    2: 443.3 * 38,   # Warping Check
    3: 696.7 * 38,   # Gumming
    4: 823.3 * 38,   # Weaving
    5: 443.3 * 38    # Weaving Check
}

# --- Tablo 4: Stok (envanter) maliyetleri (USD/m→TL/m) ---
CA_TL = {
    1: 0.013 * 38,   # Warping
    2: 0.013 * 38,   # Warping Check
    3: 0.021 * 38,   # Gumming
    4: 0.025 * 38,   # Weaving
    5: 0.025 * 38    # Weaving Check
}

# --- Tablo 4: Depolama kapasitesi (m) ---
A_storage = {
    1: 170_000,
    2: 120_000,
    3: 30_000,
    4: 350_000,
    5: 150_000
}

# --- Tablo 3: Üretim kapasitesi (m/month) ---
PM = {
    1: 425_000,
    2: 420_000,
    3: 570_000,
    4: 490_000,
    5: 410_000  # Weaving Check: makaledeki değere uygun
}

# --- Tablo 1: İsraf (waste/shrinkage) oranları (DT) ---
DT = {
    1: {1: 0.0,   2: 0.0,  3: 0.0},
    2: {1: 0.0,   2: 0.0,  3: 0.0},
    3: {1: 0.03,  2: 0.03, 3: 0.02},
    4: {1: 0.10,  2: 0.12, 3: 0.07},
    5: {1: 0.02,  2: 0.02, 3: 0.03}
}

# --- Tablo 2: Aylık talep (bin metre) → Gerçek değer için 1000 ile çarpıyoruz ---
D = {
    1: [60*1000, 70*1000, 70*1000, 85*1000, 80*1000, 60*1000, 70*1000, 90*1000, 110*1000, 80*1000, 75*1000, 70*1000],
    2: [64*1000, 71*1000, 139*1000,115*1000,133*1000,80*1000,123*1000,99*1000,163*1000,135*1000,104*1000,84*1000],
    3: [100*1000,105*1000,166*1000,123*1000,177*1000,124*1000,172*1000,194*1000,206*1000,219*1000,140*1000,149*1000]
}

# --- Tablo 5: Eğitim gereksinimi (gün) ve yeni çalışan verimliliği (oran) ---
DE = { 1: 5, 2: 12, 3: 5, 4: 5, 5: 12 }
E = { 1: 0.70, 2: 0.80, 3: 0.70, 4: 0.60, 5: 0.80 }

# --- Tablo 3: Üretim hızı (m/saat) ---
MH = {
    1: {1: 120, 2: 120, 3: 115},
    2: {1: 390, 2: 390, 3: 350},
    3: {1: 219, 2: 219, 3: 240},
    4: {1: 157, 2: 120, 3: 125},
    5: {1: 318, 2: 318, 3: 280}
}

# --- Tablo 6: Çalışma saatleri (aylık) ---
HD = [208, 184, 200, 208, 200, 200, 216, 200, 208, 208, 192, 208]

# --- İdari maliyetler (örnek değer, TL cinsinden) ---
CAC_TL = { i: 654 * 38 for i in range(1, I+1) }
CAD_TL = { i: 654 * 38 for i in range(1, I+1) }

# --- Tablo 4: Başlangıç stokları (m) ---
init_inv = {
    1: {1: 20000, 2: 10000, 3: 15000},
    2: {1: 10000, 2: 5000,  3: 7500},
    3: {1: 30000, 2: 15000, 3: 22500},
    4: {1: 40000, 2: 20000, 3: 30000},
    5: {1: 20000, 2: 10000, 3: 15000}
}

# --- Tablo 5: Başlangıç çalışan sayıları ---
init_workers = {1: 13, 2: 3, 3: 4, 4: 11, 5: 3}

# ---------------------------
# 2. LP Modelinin Kurulması
# ---------------------------
model = pulp.LpProblem("LIPROTEX_TL", pulp.LpMinimize)

# Karar Değişkenleri
P = pulp.LpVariable.dicts("P", ((l, i, t) for l in range(1, L+1)
                                  for i in range(1, I+1)
                                  for t in range(1, T+1)), lowBound=0, cat='Continuous')
H_var = pulp.LpVariable.dicts("H", ((l, i, t) for l in range(1, L+1)
                                      for i in range(1, I+1)
                                      for t in range(1, T+1)), lowBound=0, cat='Continuous')
# Stok (S) için t = 0 ... T
S = pulp.LpVariable.dicts("S", ((l, i, t) for l in range(1, L+1)
                                  for i in range(1, I+1)
                                  for t in range(0, T+1)), lowBound=0, cat='Continuous')
X = pulp.LpVariable.dicts("X", ((i, t) for i in range(1, I+1)
                                  for t in range(0, T+1)), lowBound=0, cat='Continuous')
XN = pulp.LpVariable.dicts("XN", ((i, t) for i in range(1, I+1)
                                    for t in range(1, T+1)), lowBound=0, cat='Continuous')
XD = pulp.LpVariable.dicts("XD", ((i, t) for i in range(1, I+1)
                                    for t in range(1, T+1)), lowBound=0, cat='Continuous')

# ---------------------------
# 3. Amaç Fonksiyonu (Toplam Maliyet – TL cinsinden)
# ---------------------------
objective_terms = []
for i in range(1, I+1):
    for t in range(1, T+1):
        objective_terms.append(CE_TL[i] * X[(i, t)])
        objective_terms.append(CAC_TL[i] * XN[(i, t)])
        objective_terms.append(CAD_TL[i] * XD[(i, t)])
        for l in range(1, L+1):
            objective_terms.append(CA_TL[i] * S[(l, i, t)])
model += pulp.lpSum(objective_terms), "ToplamMaliyet_TL"

# ---------------------------
# 4. Kısıtlar
# ---------------------------

# (a) Son Süreç (i = I) için talebin karşılanması
for l in range(1, L+1):
    model += S[(l, I, 0)] == init_inv[I][l], f"BaslangicStok_Line{l}_Proc{I}"
    for t in range(1, T+1):
        model += S[(l, I, t)] == S[(l, I, t-1)] + P[(l, I, t)] - D[l][t-1], f"TalepKarsilama_Line{l}_Ay{t}"

# (b) İşlem arası envanter kısıtları (i = 1 ... I-1)
for l in range(1, L+1):
    for i in range(1, I):
        model += S[(l, i, 0)] == init_inv[i][l], f"BaslangicStok_Line{l}_Proc{i}"
        for t in range(1, T+1):
            # Modelde: S(l,i,t) = S(l,i,t-1) + P(l,i,t) - P(l,i+1,t) / (1 - DT(i,l))
            # Bölme yerine ters çarpım kullanılıyor:
            model += S[(l, i, t)] == S[(l, i, t-1)] + P[(l, i, t)] - (P[(l, i+1, t)] * (1.0 / (1 - DT[i][l]))), \
                     f"IslemArasiEnvanter_Line{l}_Proc{i}_Ay{t}"

# (c) Depolama kapasitesi kısıtı
for i in range(1, I+1):
    for t in range(1, T+1):
        model += pulp.lpSum([S[(l, i, t)] for l in range(1, L+1)]) <= A_storage[i], f"DepolamaKapasitesi_Proc{i}_Ay{t}"

# (d) İş gücü denge kısıtı
for i in range(1, I+1):
    model += X[(i, 0)] == init_workers[i], f"BaslangicCalisan_Proc{i}"
    for t in range(1, T+1):
        model += X[(i, t)] == X[(i, t-1)] + XN[(i, t)] - XD[(i, t)], f"IsGucuDenge_Proc{i}_Ay{t}"

# (e) Üretim miktarının hesaplanması: P(l,i,t) = H(l,i,t) * MH(i,l)
for l in range(1, L+1):
    for i in range(1, I+1):
        for t in range(1, T+1):
            model += P[(l, i, t)] == H_var[(l, i, t)] * MH[i][l], f"UretimHesapla_Line{l}_Proc{i}_Ay{t}"

# (f) Üretim saatleri kısıtı:
for i in range(1, I+1):
    for t in range(1, T+1):
        model += pulp.lpSum([H_var[(l, i, t)] for l in range(1, L+1)]) == HD[t-1] * (X[(i, t-1)] + XN[(i, t)] * E[i] - XD[(i, t)]), \
                 f"UretimSaati_Proc{i}_Ay{t}"

# (g) Maksimum üretim kapasitesi kısıtı:
for i in range(1, I+1):
    for t in range(1, T+1):
        model += pulp.lpSum([P[(l, i, t)] for l in range(1, L+1)]) <= PM[i], f"MaksimumUretimKapasitesi_Proc{i}_Ay{t}"

# ---------------------------
# 5. Modelin Çözülmesi
# ---------------------------
solver = pulp.PULP_CBC_CMD(msg=True)
model.solve(solver)

# ---------------------------
# 6. ÇÖZÜM SONUÇLARI VE STREAMLIT İÇİN VERİLER
# ---------------------------

toplam_maliyet_TL = pulp.value(model.objective)
# Makaledeki hedef optimum: 424.074 USD → TL’ye çevrildiğinde: 424074 * 38 TL
hedef_toplam_maliyet_TL = 424074 * 38

def format_currency(value):
    return locale.format_string("%.2f", value, grouping=True)

# Üretim, stok ve üretim saatleri sonuçlarını toplamak
results = []
for i in range(1, I+1):
    for t in range(1, T+1):
        for l in range(1, L+1):
            results.append({
                "Ürün Hattı": f"Line {l}",
                "Süreç": f"Proc {i}",
                "Ay": t,
                "Üretim (m)": P[(l, i, t)].varValue,
                "Üretim Saati": H_var[(l, i, t)].varValue,
                "Stok (m)": S[(l, i, t)].varValue
            })
df_results = pd.DataFrame(results)

# Çalışan sayıları
worker_results = []
for i in range(1, I+1):
    for t in range(0, T+1):
        worker_results.append({
            "Süreç": f"Proc {i}",
            "Ay": t,
            "Çalışan Sayısı": X[(i, t)].varValue
        })
df_workers = pd.DataFrame(worker_results)

# Toplam üretim saatleri (süreç bazında)
production_hours = []
for i in range(1, I+1):
    for t in range(1, T+1):
        total_hours = sum(H_var[(l, i, t)].varValue for l in range(1, L+1))
        production_hours.append({
            "Süreç": f"Proc {i}",
            "Ay": t,
            "Toplam Üretim Saati": total_hours
        })
df_hours = pd.DataFrame(production_hours)

# ---------------------------
# 7. STREAMLIT ARAYÜZÜ VE İNTERAKTİF GÖRSELLEŞTİRMELER
# ---------------------------
st.set_page_config(page_title="LIPROTEX Üretim Planlaması (TL)", layout="wide")
st.markdown("<h1 style='text-align: center; color: #003366;'>LIPROTEX Üretim Planlaması – TL Cinsinden</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #006699;'>Makaledeki LP Modeline Dayalı Optimal Çözüm ve İnteraktif Görselleştirmeler</h3>", unsafe_allow_html=True)

# Sol sidebar: Filtreleme seçenekleri
st.sidebar.header("Filtreleme Seçenekleri")
selected_line = st.sidebar.selectbox("Ürün Hattı Seçin", options=[1, 2, 3], format_func=lambda x: f"Line {x}")
selected_proc = st.sidebar.selectbox("Süreç Seçin", options=[1, 2, 3, 4, 5], format_func=lambda x: f"Proc {x}")

# Üst özet bilgileri: Model durumu, toplam maliyet ve hedef maliyet
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Durumu", pulp.LpStatus[model.status])
with col2:
    st.metric("Toplam Maliyet (TL)", f"{format_currency(toplam_maliyet_TL)} TL")
with col3:
    st.metric("Hedef Maliyet (TL)", f"{format_currency(hedef_toplam_maliyet_TL)} TL")

if abs(toplam_maliyet_TL - hedef_toplam_maliyet_TL) > 10000:
    st.error("Optimum TL değeri makaledeki hedefle uyuşmuyor. Lütfen veri setleri ve ölçek dönüşümlerini kontrol ediniz.")
else:
    st.success("Optimum TL değeri makaledeki sonuçla uyumlu.")

st.markdown("---")

# Sonuçların tablosal sunumu (DataFrame)
st.subheader("Üretim, Stok ve Üretim Saatleri Sonuçları")
st.dataframe(df_results.style.format({
    "Üretim (m)": lambda x: format_currency(x) if pd.notnull(x) else "",
    "Üretim Saati": lambda x: format_currency(x) if pd.notnull(x) else "",
    "Stok (m)": lambda x: format_currency(x) if pd.notnull(x) else ""
}), height=400)

st.subheader("Çalışan Sayıları")
st.dataframe(df_workers.style.format({
    "Çalışan Sayısı": lambda x: format_currency(x) if pd.notnull(x) else ""
}), height=300)

st.subheader("Toplam Üretim Saatleri (Süreç Bazında)")
st.dataframe(df_hours.style.format({
    "Toplam Üretim Saati": lambda x: format_currency(x) if pd.notnull(x) else ""
}), height=300)

# İleri Düzey İnteraktif Grafikler
st.markdown("### İleri Düzey İnteraktif Grafikler")

# Grafik 1: Seçilen ürün hattı ve süreç için stok zaman serisi (Plotly Express)
df_filtered = df_results[(df_results["Ürün Hattı"] == f"Line {selected_line}") & 
                         (df_results["Süreç"] == f"Proc {selected_proc}")]
fig_stok = px.line(df_filtered, x="Ay", y="Stok (m)",
                   title=f"Line {selected_line} – Proc {selected_proc}: Aylık Stok Seviyesi",
                   markers=True,
                   labels={"Stok (m)": "Stok (m)", "Ay": "Ay"})
fig_stok.update_layout(template="plotly_white", title_font=dict(size=20, color="#003366"))
st.plotly_chart(fig_stok, use_container_width=True)

# Grafik 2: Ürün hatlarına göre tüm süreçlerde toplam üretim miktarı (Group Bar Chart)
total_production = []
for l in range(1, L+1):
    for t in range(1, T+1):
        total_prod = sum(P[(l, i, t)].varValue for i in range(1, I+1))
        total_production.append({"Ürün Hattı": f"Line {l}", "Ay": t, "Üretim (m)": total_prod})
df_total_prod = pd.DataFrame(total_production)
fig_prod = px.bar(df_total_prod, x="Ay", y="Üretim (m)", color="Ürün Hattı", barmode="group",
                  title="Ürün Hatlarına Göre Aylık Toplam Üretim Miktarları",
                  labels={"Üretim (m)": "Üretim (m)", "Ay": "Ay"})
fig_prod.update_layout(template="plotly_white", title_font=dict(size=20, color="#003366"))
st.plotly_chart(fig_prod, use_container_width=True)

# Grafik 3: Süreç bazında çalışan sayıları zaman serisi (Plotly Graph Objects)
fig_workers = go.Figure()
for i in range(1, I+1):
    df_temp = df_workers[df_workers["Süreç"] == f"Proc {i}"]
    fig_workers.add_trace(go.Scatter(x=df_temp["Ay"], y=df_temp["Çalışan Sayısı"],
                                     mode='lines+markers', name=f"Proc {i}",
                                     hovertemplate="Ay: %{x}<br>Çalışan: %{y:,.2f}"))
fig_workers.update_layout(
    title="Süreç Bazında Çalışan Sayıları Zaman Serisi",
    xaxis_title="Ay",
    yaxis_title="Çalışan Sayısı",
    template="plotly_white",
    title_font=dict(size=20, color="#003366")
)
st.plotly_chart(fig_workers, use_container_width=True)

# Ek Görselleştirme: Model Çözüm Özeti (Metin ve özel CSS ile)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #003366;'>Model Çözüm Özeti</h3>", unsafe_allow_html=True)
summary_html = f"""
<div style="display: flex; justify-content: space-around; font-size: 18px; color: #004466;">
  <div><strong>Model Durumu:</strong> {pulp.LpStatus[model.status]}</div>
  <div><strong>Toplam Maliyet (TL):</strong> {format_currency(toplam_maliyet_TL)} TL</div>
  <div><strong>Hedef Maliyet (TL):</strong> {format_currency(hedef_toplam_maliyet_TL)} TL</div>
</div>
"""
st.markdown(summary_html, unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 14px;'>Tüm veriler makaledeki tablolar esas alınarak TL cinsinden (38 TL/USD kuru uygulanarak) hesaplanmıştır.</p>", unsafe_allow_html=True)

# ---------------------------
# 8. Son Mesaj
# ---------------------------
st.markdown("<h4 style='text-align: center; color: #006600;'>Model çözümü ve görselleştirmeler tamamlandı.</h4>", unsafe_allow_html=True)

