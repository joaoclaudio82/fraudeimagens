from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from fraud_detector import AnalysisConfig, analyze_image


st.set_page_config(page_title="ImageGuard PoC", page_icon="🔎", layout="wide")
st.title("ImageGuard — triagem de risco em comprovantes")
st.caption("Protótipo explicável para OCR, qualidade, duplicidade e sinais técnicos de edição.")

if "hashes" not in st.session_state:
    st.session_state.hashes = []
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Dados operacionais esperados")
    order_id = st.text_input("Número do pedido")
    date = st.text_input("Data")
    recipient = st.text_input("Nome do destinatário")
    code = st.text_input("Código ou referência")
    st.divider()
    st.header("Limiares")
    review_threshold = st.slider("Revisão humana a partir de", 20, 70, 35)
    blur_threshold = st.slider("Nitidez mínima", 20, 200, 80)
    if st.button("Limpar histórico da sessão"):
        st.session_state.hashes = []
        st.session_state.history = []
        st.rerun()

uploaded = st.file_uploader("Envie um comprovante (JPG, JPEG ou PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    data = uploaded.getvalue()
    expected = {"pedido": order_id, "data": date, "destinatário": recipient, "código": code}
    # A operação ajusta a sensibilidade da fila pela configuração; a decisão é calculada
    # em um único lugar (fraud_detector.scoring) e fica registrada no relatório.
    config = AnalysisConfig(blur_warn_variance=float(blur_threshold), review_threshold=int(review_threshold))
    try:
        result = analyze_image(data, uploaded.name, expected, st.session_state.hashes, config)
    except Exception as exc:
        st.error(f"Não foi possível analisar a imagem: {exc}")
        st.stop()

    score = result["risk_score"]
    ela_render = result.pop("_ela_image")
    result.pop("_images", None)

    left, center, right = st.columns([1.3, 1, 1])
    left.image(data, caption="Imagem recebida", use_container_width=True)
    center.image(ela_render, caption="Mapa ELA (áreas claras merecem inspeção)", use_container_width=True)
    right.metric("Score de risco", f"{score}/100")
    right.metric("Encaminhamento", result["decision"])
    right.write(result["disclaimer"])

    if result["signal_errors"]:
        st.warning("Alguns sinais falharam e foram ignorados: " + ", ".join(
            f"{name} ({error})" for name, error in result["signal_errors"].items()))

    st.subheader("Evidências")
    if result["findings"]:
        st.dataframe(pd.DataFrame(result["findings"]), use_container_width=True, hide_index=True)
    else:
        st.success("Nenhum indicador configurado foi acionado.")

    tab_ocr, tab_meta, tab_audit = st.tabs(["Texto extraído", "Metadados", "Registro de auditoria"])
    with tab_ocr:
        if result["ocr_warning"]:
            st.warning(result["ocr_warning"])
        st.text_area("OCR", result["ocr_text"] or "Nenhum texto reconhecido.", height=220)
        if result["field_matches"]:
            st.json(result["field_matches"])
    with tab_meta:
        st.json({"formato": result["format"], "dimensões": result["dimensions"], "exif": result["exif"]})
    with tab_audit:
        audit_json = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        st.code(audit_json, language="json")
        st.download_button("Baixar relatório JSON", audit_json,
                           file_name=f"auditoria-{datetime.now():%Y%m%d-%H%M%S}.json",
                           mime="application/json")

    if st.button("Registrar análise no histórico da sessão", type="primary"):
        st.session_state.hashes.append(result["perceptual_hash"])
        st.session_state.history.append({
            "arquivo": result["filename"], "score": score, "decisão": result["decision"],
            "sha256": result["sha256"], "analisado_em": result["analyzed_at"]
        })
        st.success("Análise registrada. Próximos uploads serão comparados por similaridade.")

if st.session_state.history:
    st.divider()
    st.subheader("Histórico da sessão")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
