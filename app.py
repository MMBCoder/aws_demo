# app.py (updated)
import os
import json
import re
import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
import importlib

# try to import agents.py (recommended). If missing, fallback to internal simple implementations.
AGENTS_PATH = "/home/sagemaker-user/AI-COPS"
if AGENTS_PATH not in os.sys.path:
    os.sys.path.append(AGENTS_PATH)

try:
    import agents
    importlib.reload(agents)
    HAVE_AGENTS = True
except Exception:
    agents = None
    HAVE_AGENTS = False

# Optional Bedrock (boto3) usage
try:
    import boto3
except Exception:
    boto3 = None

# ---------- CONFIG (edit if needed) ----------
KB_DIR = Path(os.environ.get("KB_DIR", "/home/sagemaker-user/AI-COPS/data"))
SEG_FILE = KB_DIR / "Segmentation_requirements.txt"
CAMPAIGN_FILE = KB_DIR / "campaign_requirements.txt"
CODE_KB_FILE = KB_DIR / "code_kb.txt"
LOG_FILE = KB_DIR / "agent_interaction_log.csv"
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "")  # model id to invoke
# ------------------------------------------------

st.set_page_config(page_title="Campaign Multi-Agent Chat", layout="wide")

# ---------- Utilities ----------
def ensure_paths():
    KB_DIR.mkdir(parents=True, exist_ok=True)
    # create header if file missing or empty
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        LOG_FILE.write_text("timestamp,actor,role,message,wfn,action,notes\n", encoding="utf-8")

def read_kb_table(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="|", dtype=str).fillna("")

def load_code_kb():
    if not CODE_KB_FILE.exists():
        return ""
    return CODE_KB_FILE.read_text(encoding="utf-8")

def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def append_log(actor, role, message, wfn="", action="", notes=""):
    """
    Robust append to CSV log. Handles empty/malformed files.
    """
    ensure_paths()
    ts = _now_iso()
    row = {
        "timestamp": ts,
        "actor": actor,
        "role": role,
        "message": str(message).replace("\n"," ").replace(",",";"),
        "wfn": wfn,
        "action": action,
        "notes": str(notes).replace("\n"," ").replace(",",";")
    }
    expected_cols = ["timestamp","actor","role","message","wfn","action","notes"]
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
            try:
                df = pd.read_csv(LOG_FILE)
                if df.columns.size == 0:
                    df = pd.DataFrame(columns=expected_cols)
                for c in expected_cols:
                    if c not in df.columns:
                        df[c] = ""
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame([row], columns=expected_cols)
            except Exception:
                # recreate file if unreadable
                df = pd.DataFrame([row], columns=expected_cols)
        else:
            df = pd.DataFrame([row], columns=expected_cols)
        df.to_csv(LOG_FILE, index=False)
    except Exception:
        # last-resort plain append
        try:
            if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
                with open(LOG_FILE, "w", encoding="utf-8") as f:
                    f.write(",".join(expected_cols) + "\n")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                vals = [str(row[c]).replace(",", ";") for c in expected_cols]
                f.write(",".join(vals) + "\n")
        except Exception as e:
            print("Failed to write log:", e)

# ---------- Bedrock LLM wrapper ----------
def bedrock_invoke(prompt, model_id=BEDROCK_MODEL_ID, region=BEDROCK_REGION, timeout=60):
    """
    Generic Bedrock invocation wrapper.
    """
    if not boto3:
        raise RuntimeError("boto3 not available in environment.")
    if not model_id:
        raise RuntimeError("BEDROCK_MODEL_ID not set.")
    client = boto3.client("bedrock-runtime", region_name=region) if region else boto3.client("bedrock-runtime")
    body = json.dumps({"input": prompt}).encode("utf-8")
    response = client.invoke_model(body=body, contentType="application/json", accept="application/json", modelId=model_id)
    resp_bytes = response.get("body").read() if hasattr(response.get("body"), "read") else response.get("body")
    try:
        resp_json = json.loads(resp_bytes)
        for key in ("output","content","generated_text","text"):
            if key in resp_json:
                return resp_json[key]
        for v in resp_json.values():
            if isinstance(v, str):
                return v
        return str(resp_json)
    except Exception:
        try:
            return resp_bytes.decode("utf-8")
        except Exception:
            return str(resp_bytes)

# ---------- Local fallback LLM functions ----------
def local_summarize_campaign(campaign_row, segments_df):
    """Deterministic local summary of campaign and its segments"""
    lines = []
    lines.append(f"Campaign: {campaign_row.get('Campaign_Name', campaign_row.get('Client','Unknown'))}")
    for k in ("WFN","Client","Target_Criteria","Channel"):
        if campaign_row.get(k):
            lines.append(f"{k}: {campaign_row.get(k)}")
    lines.append("Segments:")
    for _, r in segments_df.iterrows():
        lines.append(f" - {r.to_dict()}")
    return "\n".join(lines)

def llm_safe_invoke(prompt, role="assistant"):
    """
    Attempts to call Bedrock; if unsuccessful, returns a deterministic fallback.
    """
    try:
        if boto3 and BEDROCK_MODEL_ID:
            return bedrock_invoke(prompt)
    except Exception as e:
        append_log("system", "error", f"bedrock invoke failed: {e}", action="bedrock_error", notes=str(e))
    # fallback:
    return "[LLM fallback] " + (prompt[:1000] if prompt else "(empty)")

# ---------- Helper: normalize client name ----------
def normalize_client_name(raw: str) -> str:
    if not raw:
        return "CLIENT"
    name = str(raw).strip()
    name = re.sub(r"[^0-9A-Za-z]+", "_", name)
    name = name.strip("_")
    return name if name else "CLIENT"

# ---------- Agent wrappers (prefer agents.py when available) ----------
def campaign_requirements_agent_llm(wfn):
    if HAVE_AGENTS:
        return agents.campaign_requirements_agent(wfn)
    # fallback local
    camp_df = read_kb_table(CAMPAIGN_FILE)
    seg_df = read_kb_table(SEG_FILE)
    row = camp_df[camp_df['WFN'].astype(str).str.strip()==str(wfn).strip()]
    if row.empty:
        msg = f"No campaign found for WFN {wfn}."
        append_log("campaign_agent","info", msg, wfn=wfn, action="not_found")
        return {"found": False, "message": msg}
    campaign = row.iloc[0].to_dict()
    segments = seg_df[seg_df['WFN'].astype(str).str.strip()==str(wfn).strip()].reset_index(drop=True)
    prompt = f"""Summarize campaign row and segments for WFN {wfn}."""
    resp = llm_safe_invoke(prompt)
    try:
        parsed = json.loads(resp)
        summary = parsed.get("summary", local_summarize_campaign(campaign, segments))
        confirm_prompt = parsed.get("confirm_prompt","Are these requirements correct? (Yes/No)")
    except Exception:
        summary = local_summarize_campaign(campaign, segments)
        confirm_prompt = "Are these requirements correct? (Yes/No)"
    append_log("campaign_agent","present", summary, wfn=wfn, action="present_requirements")
    return {"found": True, "campaign": campaign, "segments": segments, "summary": summary, "confirm_prompt": confirm_prompt}

def code_generation_agent_llm(campaign, segments, wfn):
    if HAVE_AGENTS:
        # agents.code_generation_agent returns {'sas':..., 'export_name':...}
        return agents.code_generation_agent(campaign, segments, wfn)
    # fallback local generation + strict client export naming
    template = load_code_kb()
    client_norm = normalize_client_name(campaign.get("Client", campaign.get("client","")))
    expected_export_name = f"{client_norm}_EM_seeds"
    prompt = f"""Generate SAS for WFN {wfn}. REQUIRED: export dataset must be named exactly {expected_export_name}."""
    sas_text = llm_safe_invoke(prompt)
    # If LLM fallback returned the prompt token (meaning no real SAS), do deterministic replace
    if sas_text.startswith("[LLM fallback]"):
        sas = re.sub(r'(%let\s+target_wfn\s*=\s*)\w+(\s*;)', rf"\1{wfn}\2", template, flags=re.IGNORECASE)
        sas = sas.replace("WGRN_EM_seeds", expected_export_name)
        header = f"/* Fallback SAS - client: {client_norm} - wfn: {wfn} */\n"
        sas = header + sas
        append_log("code_agent","generated_fallback", sas[:1000], wfn=wfn, action="generate_code_fallback")
        return {"sas": sas, "export_name": expected_export_name}
    # otherwise we got some generated text — ensure naming and macros corrected
    sas_text = re.sub(r'(%let\s+target_wfn\s*=\s*)\w+(\s*;)', rf"\1{wfn}\2", sas_text, flags=re.IGNORECASE)
    sas_text = sas_text.replace("WGRN_EM_seeds", expected_export_name)
    append_log("code_agent","generated", (sas_text or "")[:1000], wfn=wfn, action="generate_code")
    return {"sas": sas_text, "export_name": expected_export_name}

def code_audit_agent_llm(sas_text, campaign, segments, max_iterations=3):
    if HAVE_AGENTS:
        return agents.code_audit_agent(sas_text, campaign, segments, max_iterations=max_iterations)
    # fallback heuristics
    client_norm = normalize_client_name(campaign.get("Client", campaign.get("client","")))
    expected_export = f"{client_norm}_EM_seeds"
    lower = (sas_text or "").lower()
    issues = []
    recs = []
    if "segmentation_requirements" not in lower and "segmentation" not in lower:
        issues.append("Segmentation file read/filter step not found")
        recs.append("Ensure segmentation file is read and filters applied")
    if "call streaminit" not in lower and "rand(" not in lower:
        issues.append("Random test/control assignment not detected")
        recs.append("Add call streaminit and rand('uniform') assignment")
    if expected_export.lower() not in lower:
        issues.append(f"Export dataset name must be {expected_export}")
        recs.append(f"Replace WGRN_EM_seeds with {expected_export}")
    passed = len(issues) == 0
    append_log("audit_agent","audit_fallback", json.dumps({"passed":passed,"issues":issues})[:1000], wfn=campaign.get("WFN",""), action="audit_fallback")
    return {"passed": passed, "issues": issues, "recommendations": recs, "fixed": None}

# ---------- Streamlit formatted renderer ----------
def render_orchestration_result(res: dict):
    """Streamlit renderer for orchestration results in a structured, readable UI."""
    st.markdown("## ✅ Orchestration Result Summary")
    if not isinstance(res, dict):
        st.error("Invalid orchestration result received.")
        return
    status = res.get("status", "")
    path = res.get("path", "")
    iterations = res.get("iterations", "")
    st.markdown("### 📌 Status")
    st.write(pd.DataFrame([["Status", status], ["Saved SAS Path", path], ["Iterations", iterations]], columns=["Item", "Value"]))
    st.markdown("### 🔍 Audit Results")
    audit = res.get("audit", {})
    if isinstance(audit, dict):
        issues = audit.get("issues", [])
        recs = audit.get("recommendations", [])
        passed = audit.get("passed", False)
        audit_df = pd.DataFrame([["Passed?", "✅ YES" if passed else "❌ NO"], ["Issues", "; ".join(issues) if issues else "None"], ["Recommendations", "; ".join(recs) if recs else "None"]], columns=["Audit Item", "Value"])
        st.table(audit_df)
    else:
        st.write("Audit response:", audit)
    sas_text = res.get("sas", "")
    if sas_text:
        st.markdown("### 📄 Generated SAS Program")
        sas_text_clean = sas_text.replace("\r\n", "\n").replace("\r", "\n")
        lines = sas_text_clean.split("\n")
        n_lines = len(lines)
        n_chars = len(sas_text_clean)
        first_50 = "\n".join(lines[:50])
        sas_stats_df = pd.DataFrame([["Total Lines", n_lines], ["Characters", n_chars], ["Preview (first 50 lines)", first_50[:300]]], columns=["Item", "Value"])
        st.write(sas_stats_df)
        with st.expander("📜 Click to expand full SAS program"):
            st.code(sas_text_clean, language="sas")
        # save a preview
        save_path = KB_DIR / "sas_preview_streamlit.txt"
        save_path.write_text(first_50 + "\n\n... truncated ...\n", encoding="utf-8")
        st.info(f"SAS preview saved to: `{save_path}`")
    else:
        st.warning("No SAS program was generated.")
    with st.expander("🛠 Debug Raw Response"):
        st.json(res)

# ---------- Streamlit UI ----------
ensure_paths()

st.title("Campaign Execution — Multi Agent Chat (Bedrock-enabled)")

# Session state defaults
if "chat" not in st.session_state:
    st.session_state.chat = []
if "wfn" not in st.session_state:
    st.session_state.wfn = ""
if "campaign_ok" not in st.session_state:
    st.session_state.campaign_ok = False
if "sas_text" not in st.session_state:
    st.session_state.sas_text = None
if "audit" not in st.session_state:
    st.session_state.audit = None
if "iterations" not in st.session_state:
    st.session_state.iterations = 0

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Controls")
    wfn = st.text_input("WFN (Workfront Number)", value=st.session_state.get("wfn",""))
    if st.button("Load & Summarize Requirements"):
        st.session_state.wfn = wfn
        append_log("user","action", f"load_requirements for {wfn}", wfn=wfn, action="load")
        camp_res = campaign_requirements_agent_llm(wfn)
        if not camp_res.get("found"):
            st.warning(camp_res.get("message"))
            st.session_state.campaign_ok = False
        else:
            summary = camp_res.get("summary")
            st.session_state.chat.append({"from":"campaign_agent","role":"agent","text": summary})
            st.session_state.chat.append({"from":"campaign_agent","role":"question","text": camp_res.get("confirm_prompt","Are these correct? (Yes/No)")})
            st.session_state.campaign = camp_res["campaign"]
            st.session_state.segments = camp_res["segments"]
            st.session_state.campaign_ok = False

    if st.session_state.chat:
        st.markdown("### Conversation (latest messages)")
        for msg in st.session_state.chat[-8:]:
            who = msg["from"]
            st.write(f"**{who}**: {msg['text']}")

    if st.session_state.chat and any(m for m in st.session_state.chat if m.get("role")=="question"):
        answer = st.radio("Answer the agent question", ("Yes","No"), index=0, key="req_answer")
        if st.button("Submit Answer to Agent"):
            append_log("user","response", answer, wfn=st.session_state.wfn, action="approve_requirements")
            if answer.lower().startswith("y"):
                st.success("Requirements approved by user.")
                st.session_state.campaign_ok = True
                st.session_state.chat.append({"from":"user","role":"response","text":"Yes - requirements approved"})
            else:
                st.session_state.campaign_ok = False
                st.session_state.chat.append({"from":"user","role":"response","text":"No - requirements rejected"})
                st.warning("You rejected the requirements. Update KB files and reload.")

with col2:
    st.subheader("Code & Audit")
    if st.session_state.get("campaign_ok"):
        st.write("Campaign approved — you can generate code.")
        if st.button("Generate SAS Code (LLM)"):
            append_log("user","action","generate_code", wfn=st.session_state.wfn, action="generate")
            cg = code_generation_agent_llm(st.session_state.campaign, st.session_state.segments, st.session_state.wfn)
            st.session_state.sas_text = cg.get("sas")
            st.session_state.chat.append({"from":"code_agent","role":"agent","text":"Generated SAS code (preview below)."})
            append_log("code_agent","present", (st.session_state.sas_text or "")[:1000], wfn=st.session_state.wfn, action="present_code")

        if st.session_state.sas_text:
            st.markdown("### SAS Preview (first 3000 chars)")
            st.code(st.session_state.sas_text[:3000])
            if st.button("Run Audit (LLM)"):
                append_log("user","action","audit_request", wfn=st.session_state.wfn, action="audit")
                audit_res = code_audit_agent_llm(st.session_state.sas_text, st.session_state.campaign, st.session_state.segments)
                st.session_state.audit = audit_res
                st.session_state.iterations = 0
                if audit_res.get("passed"):
                    st.success("Audit passed")
                    st.session_state.chat.append({"from":"audit_agent","role":"agent","text":"Audit passed"})
                else:
                    st.error("Audit found issues")
                    st.session_state.chat.append({"from":"audit_agent","role":"agent","text":"Audit found issues: " + "; ".join(audit_res.get("issues",[]))})
                    for rec in audit_res.get("recommendations",[]):
                        st.write("- ", rec)
            # Approve or request refinement
            approve = st.radio("Approve final SAS code?", ("Approve","Request refinement"), key="code_approve_radio")
            if approve == "Approve" and st.button("Submit Code Approval"):
                append_log("user","response","approve_code", wfn=st.session_state.wfn, action="approve_code")
                st.success("Code approved and saved.")
                out_path = KB_DIR / f"generated_sas_{st.session_state.wfn}.sas"
                out_path.write_text(st.session_state.sas_text, encoding="utf-8")
                append_log("code_agent","saved", f"Saved to {out_path}", wfn=st.session_state.wfn, action="save_code")
                # render nicely
                render_orchestration_result({"status":"completed","path":str(out_path),"iterations":st.session_state.iterations,"audit":st.session_state.audit or {}, "sas": st.session_state.sas_text})
            elif approve == "Request refinement" and st.button("Submit Refinement Request"):
                feedback = st.text_area("Describe the changes you want (free text)", key="refine_feedback")
                if not feedback:
                    st.warning("Please enter refinement feedback in the text area first.")
                else:
                    append_log("user","feedback", feedback[:1000], wfn=st.session_state.wfn, action="refinement_requested")
                    if st.session_state.iterations < 3:
                        st.session_state.sas_text = "/* User feedback: " + feedback + " */\n" + st.session_state.sas_text
                        st.session_state.iterations += 1
                        st.success(f"Refinement attempt #{st.session_state.iterations} applied.")
                        append_log("code_agent","refined", f"Attempt #{st.session_state.iterations}", wfn=st.session_state.wfn, action="auto_refine")
                        audit_res = code_audit_agent_llm(st.session_state.sas_text, st.session_state.campaign, st.session_state.segments)
                        st.session_state.audit = audit_res
                        if audit_res.get("passed"):
                            st.success("Audit passed after refinement.")
                        else:
                            st.error("Audit still has issues: " + "; ".join(audit_res.get("issues",[])))
                    else:
                        st.warning("Maximum 3 refinements reached. Please edit KB files or contact admin.")

# ---------- show logs and KB inspection ----------
st.sidebar.header("Data & Logs")
if st.sidebar.button("Show recent log entries"):
    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE)
        st.sidebar.dataframe(df.tail(40))
    else:
        st.sidebar.write("No log file yet.")

if st.sidebar.button("Show campaign KB (preview)"):
    dfc = read_kb_table(CAMPAIGN_FILE)
    st.sidebar.dataframe(dfc.head(20))

if st.sidebar.button("Show segmentation KB (preview)"):
    dfs = read_kb_table(SEG_FILE)
    st.sidebar.dataframe(dfs.head(40))

st.sidebar.markdown("---")
st.sidebar.write(f"KB dir: {KB_DIR}")
st.sidebar.write(f"Bedrock model: {BEDROCK_MODEL_ID or 'Not configured'}")
