# app.py - Full updated file (stateful campaign approval, segmentation matrix, LLM translation, robust logging)
import os
import json
import re
import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
import importlib

# Try to import agents.py (recommended). If missing, fallback to internal implementations.
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
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
        LOG_FILE.write_text("timestamp,actor,role,message,wfn,action,notes\n", encoding="utf-8")

def read_kb_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="|", dtype=str).fillna("")

def load_code_kb() -> str:
    if not CODE_KB_FILE.exists():
        return ""
    return CODE_KB_FILE.read_text(encoding="utf-8")

def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def append_log(actor: str, role: str, message: str, wfn: str = "", action: str = "", notes: str = ""):
    """
    Robust append to CSV log. Handles empty/malformed files.
    """
    ensure_paths()
    ts = _now_iso()
    row = {
        "timestamp": ts,
        "actor": actor,
        "role": role,
        "message": str(message).replace("\n", " ").replace(",", ";"),
        "wfn": wfn,
        "action": action,
        "notes": str(notes).replace("\n", " ").replace(",", ";")
    }
    expected_cols = ["timestamp", "actor", "role", "message", "wfn", "action", "notes"]
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
            try:
                df = pd.read_csv(LOG_FILE)
                # ensure expected cols exist
                for c in expected_cols:
                    if c not in df.columns:
                        df[c] = ""
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame([row], columns=expected_cols)
            except Exception:
                df = pd.DataFrame([row], columns=expected_cols)
        else:
            df = pd.DataFrame([row], columns=expected_cols)
        df.to_csv(LOG_FILE, index=False)
    except Exception:
        # fallback file append
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
def bedrock_invoke(prompt: str, model_id: str = BEDROCK_MODEL_ID, region: str = BEDROCK_REGION, timeout: int = 60) -> str:
    """
    Generic Bedrock invocation wrapper. Returns model text or raises.
    """
    if not boto3:
        raise RuntimeError("boto3 not available in environment.")
    if not model_id:
        raise RuntimeError("BEDROCK_MODEL_ID not set.")
    client = boto3.client("bedrock-runtime", region_name=region) if region else boto3.client("bedrock-runtime")
    body = json.dumps({"input": prompt}).encode("utf-8")
    response = client.invoke_model(body=body, contentType="application/json", accept="application/json", modelId=model_id)
    resp_body = response.get("body")
    resp_bytes = resp_body.read() if hasattr(resp_body, "read") else resp_body
    try:
        resp_json = json.loads(resp_bytes)
        for key in ("output", "content", "generated_text", "text"):
            if key in resp_json:
                return resp_json[key]
        # return first string-like value
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
def local_summarize_campaign(campaign_row: dict, segments_df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Campaign: {campaign_row.get('Campaign_Name', campaign_row.get('Client', 'Unknown'))}")
    for k in ("WFN", "Client", "Target_Criteria", "Channel"):
        if campaign_row.get(k):
            lines.append(f"{k}: {campaign_row.get(k)}")
    lines.append("Segments:")
    if isinstance(segments_df, pd.DataFrame) and not segments_df.empty:
        for _, r in segments_df.iterrows():
            lines.append(f" - {r.to_dict()}")
    return "\n".join(lines)

def llm_safe_invoke(prompt: str, role: str = "assistant") -> str:
    """
    Attempts to call Bedrock; if unsuccessful, returns deterministic fallback (for offline testing).
    """
    try:
        if boto3 and BEDROCK_MODEL_ID:
            return bedrock_invoke(prompt)
    except Exception as e:
        append_log("system", "error", f"bedrock invoke failed: {e}", action="bedrock_error", notes=str(e))
    # fallback deterministic reply
    return "[LLM fallback] " + (prompt[:1000] if prompt else "(empty)")

# ---------- Helper: normalize client name ----------
def normalize_client_name(raw: str) -> str:
    if not raw:
        return "CLIENT"
    name = str(raw).strip()
    name = re.sub(r"[^0-9A-Za-z]+", "_", name)
    name = name.strip("_")
    return name if name else "CLIENT"

# ---------- Agent wrapper: campaign_requirements (fixed) ----------
def campaign_requirements_agent_llm(wfn: str) -> dict:
    """
    Return dict: { found: bool, campaign: {...}, segments: DataFrame, summary: str, confirm_prompt: str }
    """
    # prefer agents.py if it provides the function
    try:
        if HAVE_AGENTS and hasattr(agents, "campaign_requirements_agent"):
            res = agents.campaign_requirements_agent(wfn)
            return res
    except Exception as e:
        append_log("campaign_agent", "warning", f"agents.campaign_requirements_agent failed: {e}", wfn=wfn, action="agents_fallback", notes=str(e))

    # fallback: read KBs
    camp_df = read_kb_table(CAMPAIGN_FILE)
    seg_df = read_kb_table(SEG_FILE)
    row = camp_df[camp_df['WFN'].astype(str).str.strip() == str(wfn).strip()] if not camp_df.empty else pd.DataFrame()
    if row.empty:
        msg = f"No campaign found for WFN {wfn}."
        append_log("campaign_agent", "info", msg, wfn=wfn, action="not_found")
        return {"found": False, "message": msg}
    campaign = row.iloc[0].to_dict()
    segments = seg_df[seg_df['WFN'].astype(str).str.strip() == str(wfn).strip()].reset_index(drop=True) if not seg_df.empty else pd.DataFrame()

    # deterministic summary and optional LLM-enhanced summary
    summary = local_summarize_campaign(campaign, segments)
    confirm_prompt = "Are these requirements correct? (Yes/No)"

    prompt = f"""Summarize the campaign and segmentation for WFN {wfn}. Return JSON with keys 'summary' and 'confirm_prompt'.
Campaign row:
{json.dumps(campaign, indent=2)}
Segments (first 40 rows):
{segments.head(40).to_json(orient='records', force_ascii=False)}
"""
    try:
        resp = llm_safe_invoke(prompt)
        try:
            parsed = json.loads(resp)
            summary = parsed.get("summary", summary)
            confirm_prompt = parsed.get("confirm_prompt", confirm_prompt)
        except Exception:
            # ignore parsing errors; keep deterministic summary
            pass
    except Exception as e:
        append_log("campaign_agent", "warning", f"LLM summary failed: {e}", wfn=wfn, action="llm_summary_fail", notes=str(e))

    append_log("campaign_agent", "present", summary, wfn=wfn, action="present_requirements")
    return {"found": True, "campaign": campaign, "segments": segments, "summary": summary, "confirm_prompt": confirm_prompt}

# ---------- Agent wrapper: code generation & audit (prefer agents.py if present) ----------
def code_generation_agent_llm(campaign: dict, segments: pd.DataFrame, wfn: str) -> dict:
    if HAVE_AGENTS and hasattr(agents, "code_generation_agent"):
        return agents.code_generation_agent(campaign, segments, wfn)
    template = load_code_kb()
    client_norm = normalize_client_name(campaign.get("Client", campaign.get("client", "")))
    expected_export_name = f"{client_norm}_EM_seeds"
    prompt = f"""Generate SAS for WFN {wfn}. REQUIRED: export dataset must be named exactly {expected_export_name}."""
    sas_text = llm_safe_invoke(prompt)
    if sas_text.startswith("[LLM fallback]"):
        sas = re.sub(r'(%let\s+target_wfn\s*=\s*)\w+(\s*;)', rf"\1{wfn}\2", template, flags=re.IGNORECASE)
        sas = sas.replace("WGRN_EM_seeds", expected_export_name)
        header = f"/* Fallback SAS - client: {client_norm} - wfn: {wfn} */\n"
        sas = header + sas
        append_log("code_agent", "generated_fallback", sas[:1000], wfn=wfn, action="generate_code_fallback")
        return {"sas": sas, "export_name": expected_export_name}
    sas_text = re.sub(r'(%let\s+target_wfn\s*=\s*)\w+(\s*;)', rf"\1{wfn}\2", sas_text, flags=re.IGNORECASE)
    sas_text = sas_text.replace("WGRN_EM_seeds", expected_export_name)
    append_log("code_agent", "generated", (sas_text or "")[:1000], wfn=wfn, action="generate_code")
    return {"sas": sas_text, "export_name": expected_export_name}

def code_audit_agent_llm(sas_text: str, campaign: dict, segments: pd.DataFrame, max_iterations: int = 3) -> dict:
    if HAVE_AGENTS and hasattr(agents, "code_audit_agent"):
        return agents.code_audit_agent(sas_text, campaign, segments, max_iterations=max_iterations)
    client_norm = normalize_client_name(campaign.get("Client", campaign.get("client", "")))
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
    append_log("audit_agent", "audit_fallback", json.dumps({"passed": passed, "issues": issues})[:1000], wfn=campaign.get("WFN", ""), action="audit_fallback")
    return {"passed": passed, "issues": issues, "recommendations": recs, "fixed": None}

# ---------- Segmentation visualization helper ----------
def render_segmentation_view(segments):
    """
    Displays segments (DataFrame or list-of-dicts) as:
     - a clean dataframe with preferred column order
     - a pivot/matrix view: LANG_IND x T_C_IND with a combined summary cell
     - download button for CSV
    """
    import streamlit as _st
    import pandas as _pd

    if segments is None:
        _st.info("No segmentation rows available.")
        return

    # Accept DataFrame, list, or dict
    if isinstance(segments, list):
        try:
            seg_df = _pd.DataFrame(segments)
        except Exception:
            _st.write("Segments are in unexpected format.")
            return
    elif isinstance(segments, dict):
        seg_df = _pd.DataFrame([segments])
    elif isinstance(segments, pd.DataFrame):
        seg_df = segments.copy()
    else:
        _st.write("Unknown segments type:", type(segments))
        return

    # Normalize column names (trim spaces)
    seg_df.columns = [c.strip() for c in seg_df.columns]

    # Preferred column order and names
    preferred = ["WFN", "LANG_IND", "T_C_IND", "Split", "Channel", "COMM_CODE", "CIS MEMO"]
    display_cols = [c for c in preferred if c in seg_df.columns]
    if not display_cols:
        display_cols = list(seg_df.columns)

    _st.markdown("#### Segmentation rows (detail)")
    _st.dataframe(seg_df[display_cols].reset_index(drop=True))

    # Build a combined summary for pivot
    def make_summary(row):
        parts = []
        if pd.notna(row.get("Split")) and str(row.get("Split")).strip() != "":
            parts.append(f"Split: {row.get('Split')}")
        if pd.notna(row.get("Channel")) and str(row.get("Channel")).strip() != "":
            parts.append(f"Channel: {row.get('Channel')}")
        if pd.notna(row.get("COMM_CODE")) and str(row.get("COMM_CODE")).strip() != "":
            parts.append(f"COMM_CODE: {row.get('COMM_CODE')}")
        if pd.notna(row.get("CIS MEMO")) and str(row.get("CIS MEMO")).strip() != "":
            parts.append(f"CIS_MEMO: {row.get('CIS MEMO')}")
        return "\n".join(parts) if parts else "-"

    # Prepare pivot matrix
    try:
        pivot_df = seg_df.copy()
        pivot_df["summary"] = pivot_df.apply(lambda r: make_summary(r), axis=1)
        if "LANG_IND" in pivot_df.columns and "T_C_IND" in pivot_df.columns:
            matrix = pivot_df.pivot(index="LANG_IND", columns="T_C_IND", values="summary")
            _st.markdown("#### Segmentation matrix (language × Test/Control)")
            _st.dataframe(matrix.fillna("-"))
        else:
            _st.info("LANG_IND or T_C_IND columns not found — cannot build matrix view.")
    except Exception as e:
        _st.warning("Could not build pivot/matrix view: " + str(e))

    # Download CSV
    try:
        csv_bytes = seg_df[display_cols].to_csv(index=False).encode("utf-8")
        _st.download_button("Download segmentation CSV", data=csv_bytes, file_name=f"segments_{st.session_state.get('wfn','')}.csv", mime="text/csv")
    except Exception as e:
        _st.warning("Could not create download button: " + str(e))

# ---------- Campaign view + approval helper (fixed) ----------
def render_campaign_with_segments_and_approval(camp_res):
    """
    Stateful renderer. Returns True if approved, False if rejected, None if not submitted yet.
    """
    if not camp_res:
        st.info("No campaign loaded.")
        return None

    campaign = camp_res.get("campaign", {})
    segments = camp_res.get("segments", pd.DataFrame())
    wfn = str(campaign.get("WFN", st.session_state.get("wfn", "")))

    st.markdown("### Campaign Requirements (from KB)")

    # Show campaign details in a tidy table (key/value)
    kv = []
    keys_prefer = ["WFN", "Client", "Target_Criteria", "Channel", "EM_File_Layout", "DM_File_Layout", "Segment"]
    for k in keys_prefer:
        if k in campaign:
            kv.append([k, campaign.get(k)])
    for k, v in campaign.items():
        if k not in keys_prefer:
            kv.append([k, v])
    if kv:
        st.table(pd.DataFrame(kv, columns=["Field", "Value"]))
    else:
        st.write(campaign)

    # Translate Target_Criteria using LLM for clarity
    target_raw = campaign.get("Target_Criteria", "")
    st.markdown("**Target Criteria (raw)**")
    st.code(target_raw or "(empty)")

    if target_raw:
        prompt = f"""You are a helpful assistant. Translate the following campaign Target_Criteria into a clear, concise plain-English description that a campaign manager can approve. Return only the plain text translation (no JSON). Also list any assumptions in parentheses at the end.

Target_Criteria:
{target_raw}
"""
        llm_resp = llm_safe_invoke(prompt, role="campaign_translator")
        if isinstance(llm_resp, str) and llm_resp.startswith("[LLM fallback]"):
            translated = f"(LLM fallback) {target_raw.replace('%',' percent').replace('&',' and ')}"
            append_log("campaign_agent", "translate_fallback", translated, wfn=wfn, action="translate_target_criteria")
        else:
            translated = llm_resp
            append_log("campaign_agent", "translate", translated, wfn=wfn, action="translate_target_criteria")
        st.markdown("**Target Criteria — plain language (LLM translation)**")
        st.write(translated)
    else:
        st.info("No Target_Criteria found in campaign row.")

    # Render segmentation view
    st.markdown("---")
    st.markdown("### Segmentation")
    render_segmentation_view(segments)

    st.markdown("---")
    st.markdown("### Approve campaign requirements and segments")

    # Stateful keys per WFN
    radio_key = f"approval_radio_{wfn}"
    submit_key = f"approval_submitted_{wfn}"
    decision_key = f"approval_decision_{wfn}"

    # Initialize default state if missing (do NOT assign during widget creation)
    if radio_key not in st.session_state:
        st.session_state[radio_key] = "Not sure"

    options = ("Yes", "No", "Not sure")
    try:
        default_index = options.index(st.session_state.get(radio_key, "Not sure"))
    except ValueError:
        default_index = 2

    # Create the radio widget (do NOT assign the return into st.session_state manually)
    st.radio(
        "Are these campaign requirements and segment details correct?",
        options,
        index=default_index,
        key=radio_key
    )

    # When user clicks submit, read the value from session_state and persist decision
    if st.button("Submit approval decision", key=submit_key):
        choice = st.session_state.get(radio_key)
        if choice == "Yes":
            st.session_state[decision_key] = True
            append_log("user", "response", "approve_requirements", wfn=wfn, action="approve_requirements")
            st.success("Requirements approved. You may proceed to generate code.")
            return True
        elif choice == "No":
            st.session_state[decision_key] = False
            append_log("user", "response", "reject_requirements", wfn=wfn, action="reject_requirements")
            st.error("Requirements not approved. Please reach out to the Admin team for a data refresh.")
            append_log("system", "info", "User rejected requirements - advised to contact Admin for data refresh", wfn=wfn, action="admin_contact")
            return False
        else:
            st.session_state[decision_key] = None
            append_log("user", "response", "not_sure_requirements", wfn=wfn, action="not_sure")
            st.warning("Please confirm (Yes) to proceed or (No) to request Admin intervention.")
            return None

    # If user previously submitted decision, show it and keep UI visible
    if decision_key in st.session_state:
        prev = st.session_state[decision_key]
        if prev is True:
            st.success("Previously approved — you may proceed to generate code.")
            return True
        elif prev is False:
            st.error("Previously rejected. Please reach out to the Admin team for a data refresh.")
            return False
        else:
            return None

    # No submit yet
    return None

# ---------- Streamlit formatted renderer (for code result) ----------
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

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Controls")
    wfn_input = st.text_input("WFN (Workfront Number)", value=st.session_state.get("wfn", ""))
    # Load button stores camp_res in session_state, and we always render the campaign if present
    if st.button("Load & Summarize Requirements"):
        st.session_state.wfn = wfn_input
        append_log("user", "action", f"load_requirements for {wfn_input}", wfn=wfn_input, action="load")
        camp_res = campaign_requirements_agent_llm(wfn_input)
        st.session_state["last_camp_res"] = camp_res
        # also persist easily-used parts so code generation uses them
        st.session_state.campaign = camp_res.get("campaign")
        st.session_state.segments = camp_res.get("segments")
        st.session_state.campaign_ok = False
        if not camp_res.get("found"):
            append_log("campaign_agent", "info", f"campaign not found for {wfn_input}", wfn=wfn_input, action="not_found")

    # If we have a last loaded campaign, render it (stateful)
    if "last_camp_res" in st.session_state:
        decision = render_campaign_with_segments_and_approval(st.session_state["last_camp_res"])
        # update campaign_ok based on stored decision (if any)
        wfn_loaded = str(st.session_state.get("last_camp_res", {}).get("campaign", {}).get("WFN", st.session_state.get("wfn", "")))
        decision_key = f"approval_decision_{wfn_loaded}"
        if decision_key in st.session_state:
            st.session_state.campaign_ok = st.session_state[decision_key] is True
        else:
            st.session_state.campaign_ok = False

    # show conversation (simple)
    if st.session_state.chat:
        st.markdown("### Conversation (latest messages)")
        for msg in st.session_state.chat[-8:]:
            who = msg["from"]
            st.write(f"**{who}**: {msg['text']}")

with col2:
    st.subheader("Code & Audit")
    if st.session_state.get("campaign_ok"):
        st.write("Campaign approved — you can generate code.")
        if st.button("Generate SAS Code (LLM)"):
            append_log("user", "action", "generate_code", wfn=st.session_state.wfn, action="generate")
            cg = code_generation_agent_llm(st.session_state.campaign, st.session_state.segments, st.session_state.wfn)
            st.session_state.sas_text = cg.get("sas")
            st.session_state.chat.append({"from": "code_agent", "role": "agent", "text": "Generated SAS code (preview below)."})
            append_log("code_agent", "present", (st.session_state.sas_text or "")[:1000], wfn=st.session_state.wfn, action="present_code")

        if st.session_state.sas_text:
            st.markdown("### SAS Preview (first 3000 chars)")
            st.code(st.session_state.sas_text[:3000])
            if st.button("Run Audit (LLM)"):
                append_log("user", "action", "audit_request", wfn=st.session_state.wfn, action="audit")
                audit_res = code_audit_agent_llm(st.session_state.sas_text, st.session_state.campaign, st.session_state.segments)
                st.session_state.audit = audit_res
                st.session_state.iterations = 0
                if audit_res.get("passed"):
                    st.success("Audit passed")
                    st.session_state.chat.append({"from": "audit_agent", "role": "agent", "text": "Audit passed"})
                else:
                    st.error("Audit found issues")
                    st.session_state.chat.append({"from": "audit_agent", "role": "agent", "text": "Audit found issues: " + "; ".join(audit_res.get("issues", []))})
                    for rec in audit_res.get("recommendations", []):
                        st.write("- ", rec)

            # Approve or request refinement
            approve = st.radio("Approve final SAS code?", ("Approve", "Request refinement"), key="code_approve_radio")
            if approve == "Approve" and st.button("Submit Code Approval"):
                append_log("user", "response", "approve_code", wfn=st.session_state.wfn, action="approve_code")
                st.success("Code approved and saved.")
                out_path = KB_DIR / f"generated_sas_{st.session_state.wfn}.sas"
                out_path.write_text(st.session_state.sas_text, encoding="utf-8")
                append_log("code_agent", "saved", f"Saved to {out_path}", wfn=st.session_state.wfn, action="save_code")
                render_orchestration_result({"status": "completed", "path": str(out_path), "iterations": st.session_state.iterations, "audit": st.session_state.audit or {}, "sas": st.session_state.sas_text})
            elif approve == "Request refinement" and st.button("Submit Refinement Request"):
                feedback = st.text_area("Describe the changes you want (free text)", key="refine_feedback")
                if not feedback:
                    st.warning("Please enter refinement feedback in the text area first.")
                else:
                    append_log("user", "feedback", feedback[:1000], wfn=st.session_state.wfn, action="refinement_requested")
                    if st.session_state.iterations < 3:
                        st.session_state.sas_text = "/* User feedback: " + feedback + " */\n" + st.session_state.sas_text
                        st.session_state.iterations += 1
                        st.success(f"Refinement attempt #{st.session_state.iterations} applied.")
                        append_log("code_agent", "refined", f"Attempt #{st.session_state.iterations}", wfn=st.session_state.wfn, action="auto_refine")
                        audit_res = code_audit_agent_llm(st.session_state.sas_text, st.session_state.campaign, st.session_state.segments)
                        st.session_state.audit = audit_res
                        if audit_res.get("passed"):
                            st.success("Audit passed after refinement.")
                        else:
                            st.error("Audit still has issues: " + "; ".join(audit_res.get("issues", [])))
                    else:
                        st.warning("Maximum 3 refinements reached. Please edit KB files or contact admin.")
    else:
        st.info("Load a campaign and approve the requirements to enable code generation.")

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
