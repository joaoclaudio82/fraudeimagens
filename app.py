from __future__ import annotations

import io
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from fraud_detector import ANALYSIS_VERSION, AnalysisConfig, analyze_image
from fraud_detector.storage import AnalysisStore, ReviewConflict

DEFAULT_DB = os.environ.get("IMAGEGUARD_DB", "data/imageguard.sqlite")
DEFAULT_MODEL = os.environ.get("IMAGEGUARD_SCORING_MODEL", "models/scoring_logistic.json")
SEVERITY_COLORS = {"alto": (220, 40, 40), "médio": (240, 150, 20), "baixo": (60, 130, 220)}

st.set_page_config(page_title="ImageGuard", page_icon="🔎", layout="wide")
st.title("ImageGuard — triagem de risco em comprovantes")
st.caption(f"Versão da análise {ANALYSIS_VERSION}. O score prioriza revisão humana; não comprova fraude.")


@st.cache_resource
def get_store(path: str, retention_days: int, store_ocr_text: bool) -> AnalysisStore:
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return AnalysisStore(path, retention_days=retention_days, store_ocr_text=store_ocr_text)


def draw_regions(data: bytes, findings: list[dict]) -> Image.Image:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(2, image.width // 400)
    for item in findings:
        box = item.get("region")
        if not box:
            continue
        color = SEVERITY_COLORS.get(item.get("severity", "baixo"), (255, 255, 255))
        draw.rectangle((box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]), outline=color, width=width)
        draw.text((box["x"] + 4, max(0, box["y"] - 14)), item["code"], fill=color)
    return image


with st.sidebar:
    st.header("Dados operacionais esperados")
    expected = {
        "pedido": st.text_input("Número do pedido"),
        "data": st.text_input("Data da entrega"),
        "destinatario": st.text_input("Nome do destinatário"),
        "valor": st.text_input("Valor"),
        "codigo": st.text_input("Código ou referência"),
    }
    reference = st.text_input("Referência no sistema (opcional)", help="Identificador usado no histórico e na fila de revisão.")

    st.divider()
    st.header("Limiares")
    review_threshold = st.slider("Revisão humana a partir de", 20, 70, 35)
    attention_threshold = st.slider("Atenção a partir de", 5, 30, 15)
    blur_threshold = st.slider("Nitidez mínima", 20, 200, 80)

    st.divider()
    st.header("Sinais e score")
    ghost_enabled = st.checkbox("ELA localizado (JPEG ghost)", value=True)
    copy_move_enabled = st.checkbox("Copy-move (região duplicada)", value=True)
    model_available = Path(DEFAULT_MODEL).exists()
    scoring_choice = st.selectbox("Score", ["Aditivo (pontos por indicador)", "Máximo entre aditivo e modelo treinado"]
                                  if model_available else ["Aditivo (pontos por indicador)"])
    st.caption("Modelo treinado encontrado em models/." if model_available else "Nenhum modelo treinado em models/; usando o aditivo.")

    st.divider()
    st.header("Histórico e retenção")
    db_path = st.text_input("Banco SQLite", value=DEFAULT_DB, help="Hashes, análises e fila de revisão ficam aqui.")
    retention_days = st.number_input("Retenção (dias)", min_value=1, max_value=3650, value=90)
    store_ocr_text = st.checkbox("Guardar texto do OCR", value=False,
                                 help="O texto pode conter dados pessoais; desligado por padrão (LGPD).")

store = get_store(db_path, int(retention_days), bool(store_ocr_text))
config = AnalysisConfig(
    review_threshold=int(review_threshold), attention_threshold=int(attention_threshold),
    blur_warn_variance=float(blur_threshold), ghost_enabled=ghost_enabled, copy_move_enabled=copy_move_enabled,
    scoring_model=DEFAULT_MODEL if scoring_choice.startswith("Máximo") else None, scoring_mode="max",
)

uploaded = st.file_uploader("Envie um comprovante (JPG, JPEG ou PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    data = uploaded.getvalue()
    with st.spinner("Analisando..."):
        try:
            result = analyze_image(data, uploaded.name, {k: v for k, v in expected.items() if v}, store.hashes, config)
        except Exception as exc:
            st.error(f"Não foi possível analisar a imagem: {exc}")
            st.stop()
    images = result.pop("_images", {})
    result.pop("_ela_image", None)
    score = result["risk_score"]
    findings = result["findings"]

    top_left, top_right = st.columns([2.2, 1])
    with top_left:
        tabs = st.tabs(["Regiões apontadas", "Mapa ELA", "JPEG ghost", "Copy-move"] +
                       (["Detector profundo"] if "deep.overlay" in images else []))
        with tabs[0]:
            st.image(draw_regions(data, findings), caption="Imagem recebida com as regiões dos indicadores",
                     use_container_width=True)
        with tabs[1]:
            if "recompression.ela" in images:
                st.image(images["recompression.ela"], caption="ELA global: áreas claras merecem inspeção", use_container_width=True)
        with tabs[2]:
            if "recompression.ghost_overlay" in images:
                st.image(images["recompression.ghost_overlay"],
                         caption="Blocos sem o histórico de compressão do restante (vermelho) e regiões suspeitas", use_container_width=True)
            else:
                st.info("Sinal desligado.")
        with tabs[3]:
            if "copy_move.overlay" in images:
                st.image(images["copy_move.overlay"], caption="Origem (azul) e cópia (vermelho) de região duplicada", use_container_width=True)
            else:
                st.info("Sinal desligado.")
        if "deep.overlay" in images:
            with tabs[4]:
                st.image(images["deep.overlay"], use_container_width=True)
    with top_right:
        st.metric("Score de risco", f"{score}/100")
        st.metric("Encaminhamento", result["decision"])
        st.caption(f"Modelo de score: {result['score_model']}")
        st.write(result["disclaimer"])
        if result["signal_errors"]:
            st.warning("Sinais que falharam e foram ignorados: " + ", ".join(
                f"{name} ({error})" for name, error in result["signal_errors"].items()))
        if st.button("Registrar análise no histórico", type="primary"):
            analysis_id = store.save(result, reference or expected.get("pedido") or None)
            st.success(f"Análise #{analysis_id} registrada. Próximos envios serão comparados por hash e similaridade.")

    st.subheader("Evidências")
    if findings:
        table = pd.DataFrame([{
            "indicador": f["code"], "descrição": f["label"], "pontos": f["points"], "severidade": f["severity"],
            "região": (f"x={f['region']['x']} y={f['region']['y']} {f['region']['w']}×{f['region']['h']}" if f.get("region") else ""),
            "evidência": f["evidence"],
        } for f in findings])
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.success("Nenhum indicador configurado foi acionado.")

    if result["score_explanation"] and result["score_model"] != "additive":
        with st.expander("Como o score foi composto"):
            st.dataframe(pd.DataFrame(result["score_explanation"]), use_container_width=True, hide_index=True)
            st.json(result["score_details"])

    tab_ocr, tab_meta, tab_features, tab_audit = st.tabs(["Texto extraído", "Metadados", "Features", "Registro de auditoria"])
    with tab_ocr:
        if result["ocr_warning"]:
            st.warning(result["ocr_warning"])
        st.text_area("OCR", result["ocr_text"] or "Nenhum texto reconhecido.", height=220)
        details = result["signals"].get("ocr", {}).get("field_details") or {}
        if details:
            st.dataframe(pd.DataFrame([{"campo": k, **v} for k, v in details.items()]), use_container_width=True, hide_index=True)
    with tab_meta:
        meta = result["signals"].get("metadata", {})
        st.json({"formato": result["format"], "dimensões": result["dimensions"], "jpeg": meta.get("jpeg"),
                 "captura": meta.get("capture_datetime"), "exif": result["exif"]})
    with tab_features:
        st.dataframe(pd.DataFrame([{"feature": k, "valor": v} for k, v in result["features"].items()]),
                     use_container_width=True, hide_index=True)
    with tab_audit:
        audit_json = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        st.download_button("Baixar relatório JSON", audit_json,
                           file_name=f"auditoria-{datetime.now():%Y%m%d-%H%M%S}.json", mime="application/json")
        st.code(audit_json, language="json")

st.divider()
stats = store.stats()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Análises registradas", stats["analyses"])
c2.metric("Revisadas", stats["reviewed"])
c3.metric("Fraudes confirmadas", stats["confirmed_fraud"])
c4.metric("Taxa de reversão", "-" if stats["reversal_rate"] is None else f"{stats['reversal_rate']:.0%}",
          help="Parcela do que o sistema mandou revisar que o humano considerou legítimo.")

queue = store.queue(limit=20)
st.subheader(f"Fila de revisão humana ({len(queue)} pendente(s))")
if not queue:
    st.info("Nenhuma análise pendente com ATENÇÃO ou REVISAR.")
for item in queue:
    version_key = f"review-version:{db_path}:{item['id']}"
    st.session_state.setdefault(version_key, item["review_version"])
    with st.expander(f"#{item['id']} · score {item['score']} · {item['decision']} · {item['filename']} · ref {item['reference'] or '-'}"):
        st.write(", ".join(f"{f['code']} (+{f['points']})" for f in item["findings"]) or "sem indicadores")
        with st.form(f"review-{item['id']}"):
            reviewer = st.text_input("Revisor", key=f"rev-{item['id']}")
            note = st.text_input("Justificativa", key=f"note-{item['id']}")
            decision = st.radio("Decisão", ["confirmed_fraud", "legitimate", "inconclusive"], horizontal=True,
                                format_func=lambda v: {"confirmed_fraud": "Fraude confirmada", "legitimate": "Legítimo",
                                                       "inconclusive": "Inconclusivo"}[v], key=f"dec-{item['id']}")
            if st.form_submit_button("Registrar decisão"):
                try:
                    store.review(item["id"], decision, reviewer, note,
                                 expected_version=st.session_state[version_key])
                except ReviewConflict:
                    st.session_state.pop(version_key, None)
                    st.error("O parecer foi alterado por outro revisor. Recarregue a página antes de decidir.")
                except KeyError:
                    st.session_state.pop(version_key, None)
                    st.error("Esta análise não está mais disponível. Recarregue a página.")
                else:
                    st.session_state.pop(version_key, None)
                    st.success("Decisão registrada no histórico de pareceres.")
                    st.rerun()

history = store.list(limit=20)
if history:
    st.subheader("Histórico")
    st.dataframe(pd.DataFrame(history)[["id", "created_at", "filename", "reference", "score", "decision", "status", "reviewer"]],
                 use_container_width=True, hide_index=True)

    with st.expander("Consultar histórico de pareceres"):
        selected = st.selectbox("Análise", [row["id"] for row in history],
                                format_func=lambda value: f"Análise #{value}")
        after = st.number_input("Mostrar versões após", min_value=0, value=0, step=1)
        try:
            events = store.review_history(selected, limit=50, after_version=int(after))
        except KeyError:
            st.info("Análise não disponível. Atualize a página.")
        else:
            if events:
                st.dataframe(pd.DataFrame(events).rename(columns={
                    "version": "Versão", "status": "Parecer", "reviewer": "Revisor",
                    "note": "Justificativa", "reviewed_at": "Data (UTC)", "source": "Origem",
                }), use_container_width=True, hide_index=True)
                if len(events) == 50:
                    st.caption(f"Para continuar, informe {events[-1]['version']} em 'Mostrar versões após'.")
            else:
                st.info("Nenhum parecer nesta faixa de versões.")
