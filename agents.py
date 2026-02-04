"""
agents.py

Multi-agent utilities for Campaign Execution app.

- Path assumptions: KB files are in /home/sagemaker-user/AI-COPS/data
  - campaign_requirements.txt  (pipe-delimited)
  - Segmentation_requirements.txt (pipe-delimited)
  - code_kb.txt (SAS template)
- Log file: agent_interaction_log.csv in same data dir

Provides:
- robust logging (append, handles empty/corrupt file)
- campaign_requirements_agent (reads KB + summarizes)
- code_generation_agent (LLM-enabled with fallback; ensures <Client>_EM_seeds naming)
- code_audit_agent (LLM-enabled with fallback; enforces <Client>_EM_seeds present)
- small helper functions
"""

from pathlib import Path
import pandas as pd
import json
import re
import datetime
import os
from typing import Dict, Any, Optional
import traceback

# Optional AWS Bedrock support
try:
    import boto3
except Exception:
    boto3 = None

# ---------------------------
# Configuration (edit here)
# ---------------------------
KB_DIR = Path(os.environ.get("KB_DIR", "/home/sagemaker-user/AI-COPS/data"))
SEG_FILE = KB_DIR / "Segmentation_requirements.txt"
CAMPAIGN_FILE = KB_DIR / "campaign_requirements.txt"
CODE_KB_FILE = KB_DIR / "code_kb.txt"
LOG_FILE = KB_DIR / "agent_interaction_log.csv"

# Bedrock config via env vars (set when needed)
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "")

# Ensure data dir exists
KB_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Utilities & robust logger
# ---------------------------
def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def append_log(actor: str, role: str, message: str, wfn: str = "", action: str = "", notes: str = "") -> None:
    """
    Append a single row to the CSV log. Robust to empty or malformed existing files.
    Columns: timestamp, actor, role, message, wfn, action, notes
    """
    expected_cols = ["timestamp", "actor", "role", "message", "wfn", "action", "notes"]
    row = {
        "timestamp": _now_iso(),
        "actor": actor,
        "role": role,
        "message": (message or "").replace("\n", " ").replace(",", ";"),
        "wfn": wfn,
        "action": action,
        "notes": (notes or "").replace("\n", " ").replace(",", ";"),
    }
    try:
        # create parent if missing
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
            try:
                df = pd.read_csv(LOG_FILE)
                # if malformed (zero columns)
                if df.columns.size == 0:
                    df = pd.DataFrame(columns=expected_cols)
                # ensure expected columns exist
                for c in expected_cols:
                    if c not in df.columns:
                        df[c] = ""
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame([row], columns=expected_cols)
            except Exception as read_err:
                # fallback: recreate file
                print("Warning: failed to read existing log (will recreate). Err:", read_err)
                df = pd.DataFrame([row], columns=expected_cols)
        else:
            df = pd.DataFrame([row], columns=expected_cols)
        df.to_csv(LOG_FILE, index=False)
    except Exception as e:
        # last-resort fallback append
        try:
            if not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0:
                with open(LOG_FILE, "w", encoding="utf-8") as f:
                    f.write(",".join(expected_cols) + "\n")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                vals = [str(row[c]).replace(",", ";") for c in expected_cols]
                f.write(",".join(vals) + "\n")
        except Exception as e2:
            print("Critical: failed to append to log file:", e2)

# ---------------------------
# KB Loaders
# ---------------------------
def read_kb_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="|", dtype=str).fillna("")

def load_code_kb() -> str:
    if not CODE_KB_FILE.exists():
        return ""
    return CODE_KB_FILE.read_text(encoding="utf-8")

# ---------------------------
# Bedrock wrapper (optional)
# ---------------------------
def bedrock_invoke(prompt: str, model_id: Optional[str] = None, region: Optional[str] = None, timeout: int = 60) -> str:
    """
    Generic Bedrock invocation wrapper. Uses boto3 client('bedrock-runtime').invoke_model.
    NOTE: Model-specific request/response bodies may differ. This implements a simple body {"input": prompt}
    and attempts to parse common response shapes.
    """
    if boto3 is None:
        raise RuntimeError("boto3 is not installed in this environment.")
    model_id = model_id or BEDROCK_MODEL_ID
    if not model_id:
        raise RuntimeError("BEDROCK_MODEL_ID is not configured.")
    client = boto3.client("bedrock-runtime", region_name=region) if region else boto3.client("bedrock-runtime")
    body = json.dumps({"input": prompt}).encode("utf-8")
    resp = client.invoke_model(body=body, contentType="application/json", accept="application/json", modelId=model_id)
    raw = resp.get("body")
    # body can be a streaming object or bytes
    try:
        raw_bytes = raw.read() if hasattr(raw, "read") else raw
        text = raw_bytes.decode("utf-8") if isinstance(raw_bytes, (bytes, bytearray)) else str(raw_bytes)
        # try to load JSON
        try:
            parsed = json.loads(text)
            # common keys that might hold reply text
            for k in ("output", "content", "generated_text", "text", "message", "result"):
                if k in parsed and isinstance(parsed[k], str):
                    return parsed[k]
            # if dict values contain string, return first string
            for v in parsed.values():
                if isinstance(v, str):
                    return v
            return json.dumps(parsed)
        except Exception:
            return text
    except Exception:
        return str(raw)

# Safe invocation with fallback
def llm_safe_invoke(prompt: str) -> str:
    """
    Try Bedrock if configured; otherwise return deterministic fallback.
    Also logs Bedrock errors.
    """
    try:
        if boto3 is not None and BEDROCK_MODEL_ID:
            return bedrock_invoke(prompt, model_id=BEDROCK_MODEL_ID, region=BEDROCK_REGION)
    except Exception as e:
        append_log("system", "error", f"Bedrock invoke failed: {e}", action="bedrock_error", notes=str(e))
    # fallback deterministic simple behavior: echo short summary
    return f"[LLM fallback] {prompt[:2000]}"

# ---------------------------
# Helpers
# ---------------------------
def normalize_client_name(raw: str) -> str:
    if not raw:
        return "CLIENT"
    name = str(raw).strip()
    # replace non-alphanumeric with underscores
    name = re.sub(r"[^0-9A-Za-z]+", "_", name)
    # remove leading/trailing underscores
    name = name.strip("_")
    if name == "":
        return "CLIENT"
    return name

def local_summarize_campaign(campaign_row: Dict[str, Any], segments_df: pd.DataFrame) -> str:
    lines = []
    client = campaign_row.get("Client", campaign_row.get("client", ""))
    lines.append(f"Campaign (Client={client})")
    for k in ("WFN", "Client", "Target_Criteria", "Channel"):
        if campaign_row.get(k):
            lines.append(f"{k}: {campaign_row.get(k)}")
    lines.append("Segments:")
    for _, r in segments_df.iterrows():
        lines.append(f" - {r.to_dict()}")
    return "\n".join(lines)

# ---------------------------
# Agents
# ---------------------------
def campaign_requirements_agent(wfn: str) -> Dict[str, Any]:
    """
    Read campaign and segmentation KB, return a summary and raw rows.
    """
    try:
        camp_df = read_kb_table(CAMPAIGN_FILE)
        seg_df = read_kb_table(SEG_FILE)
        row = camp_df[camp_df['WFN'].astype(str).str.strip() == str(wfn).strip()]
        if row.empty:
            msg = f"No campaign found for WFN {wfn}"
            append_log("campaign_agent", "info", msg, wfn=wfn, action="not_found")
            return {"found": False, "message": msg}
        campaign = row.iloc[0].to_dict()
        segments = seg_df[seg_df['WFN'].astype(str).str.strip() == str(wfn).strip()].reset_index(drop=True)
        # LLM prompt to summarize (try LLM, fallback to local)
        prompt = f"""Summarize campaign and segmentation for user review.
Campaign row:
{json.dumps(campaign, indent=2)}
Segments ({len(segments)} rows):
{segments.to_json(orient='records', force_ascii=False)}
Return JSON: {{ "summary": "...", "confirm_prompt": "Are these requirements correct? (Yes/No)" }}
"""
        try:
            resp = llm_safe_invoke(prompt)
            parsed = {}
            try:
                parsed = json.loads(resp)
            except Exception:
                # LLM may return plain text — fallback to deterministic summary
                parsed = {"summary": local_summarize_campaign(campaign, segments), "confirm_prompt": "Are these requirements correct? (Yes/No)"}
            summary = parsed.get("summary", local_summarize_campaign(campaign, segments))
            confirm_prompt = parsed.get("confirm_prompt", "Are these requirements correct? (Yes/No)")
            append_log("campaign_agent", "present", summary, wfn=wfn, action="present_requirements")
            return {"found": True, "campaign": campaign, "segments": segments, "summary": summary, "confirm_prompt": confirm_prompt}
        except Exception as e:
            append_log("campaign_agent", "error", f"LLM error: {e}", wfn=wfn, action="error")
            summary = local_summarize_campaign(campaign, segments)
            return {"found": True, "campaign": campaign, "segments": segments, "summary": summary, "confirm_prompt": "Are these requirements correct? (Yes/No)"}
    except Exception as e:
        append_log("campaign_agent", "error", f"Exception: {e}", wfn=wfn, action="error", notes=traceback.format_exc())
        return {"found": False, "message": f"Exception while loading campaign: {e}"}

def code_generation_agent(campaign: Dict[str, Any], segments: pd.DataFrame, wfn: str) -> Dict[str, Any]:
    """
    Generate SAS code. Prefer LLM; fallback to deterministic template injection.
    Ensures export dataset is named <Client>_EM_seeds.
    """
    try:
        template = load_code_kb()
        client_norm = normalize_client_name(campaign.get("Client", campaign.get("client", "")))
        expected_export_name = f"{client_norm}_EM_seeds"

        # Build prompt instructing the LLM to use the exact export dataset name
        prompt = f"""You are an expert SAS developer. Using the provided SAS template, produce a complete SAS program.
MANDATORY: export dataset must be named exactly: {expected_export_name}
Replace any occurrence of 'WGRN_EM_seeds' or other placeholders with {expected_export_name}.
Ensure:
- The campaign WFN is set to {wfn}
- Target_Criteria from campaign is applied as filters
- Segmentation rows are applied
- Test/control assignment with rand/call streaminit exists
- Creation of {expected_export_name} and a write/export step exist

SAS Template (truncated to 4000 chars):
{template[:4000]}

Campaign meta:
{json.dumps(campaign, indent=2)}
Segments (first 40 rows):
{segments.to_json(orient='records', force_ascii=False)[:4000]}

Return only the final SAS program as plain text (no explanations).
"""
        try:
            sas_text = llm_safe_invoke(prompt)
            if not sas_text or "LLM fallback" in sas_text:
                raise RuntimeError("LLM returned no usable SAS.")
            # Ensure required replacements if LLM ignored instruction
            sas_text = re.sub(r'(%let\s+target_wfn\s*=\s*)\w+(\s*;)', rf"\1{wfn}\2", sas_text, flags=re.IGNORECASE)
            sas_text = sas_text.replace("WGRN_EM_seeds", expected_export_name)
            # Also replace any bare mentions 'WGRN_EM' etc defensively
            sas_text = re.sub(r"\bWGRN_EM\b", expected_export_name, sas_text, flags=re.IGNORECASE)
            append_log("code_agent", "generated", (sas_text or "")[:1000], wfn=wfn, action="generate_code")
            return {"sas": sas_text, "export_name": expected_export_name}
        except Exception as e:
            # deterministic fallback: inject header, replace macro and WGRN_EM_seeds tokens
            append_log("code_agent", "warning", f"LLM failed or returned unusable text: {e}", wfn=wfn, action="generate_fallback")
            header = f"/* Fallback generated SAS */\n/* WFN: {wfn} */\n/* CLIENT: {client_norm} */\n"
            sas = template
            sas = re.sub(r'(%let\s+target_wfn\s*=\s*)\w+(\s*;)', rf"\1{wfn}\2", sas, flags=re.IGNORECASE)
            sas = sas.replace("WGRN_EM_seeds", expected_export_name)
            sas = header + sas
            append_log("code_agent", "generated_fallback", sas[:1000], wfn=wfn, action="generate_code_fallback")
            return {"sas": sas, "export_name": expected_export_name}
    except Exception as e:
        append_log("code_agent", "error", f"Exception: {e}", wfn=wfn, action="error", notes=traceback.format_exc())
        return {"sas": "", "export_name": ""}

def code_audit_agent(sas_text: str, campaign: Dict[str, Any], segments: pd.DataFrame, max_iterations: int = 3) -> Dict[str, Any]:
    """
    Audit SAS program. Prefer LLM to produce a JSON audit; fallback to heuristics.
    Returns: { passed: bool, issues: [...], recommendations: [...], fixed_code: Optional[str] }
    """
    try:
        client_norm = normalize_client_name(campaign.get("Client", campaign.get("client", "")))
        expected_export_name = f"{client_norm}_EM_seeds"

        prompt = f"""You are a SAS code auditor. Inspect the following SAS program (truncated). Check it against the campaign metadata.
Return JSON: {{ "passed": true/false, "issues": ["..."], "recommendations": ["..."], "fixed_code": "...optional..." }}

SAS (truncated 4000 chars):
{sas_text[:4000]}

Campaign:
{json.dumps(campaign, indent=2)}

Ensure one required check: the export dataset name must be exactly {expected_export_name}.
"""
        try:
            resp = llm_safe_invoke(prompt)
            parsed = {}
            try:
                parsed = json.loads(resp)
            except Exception:
                # LLM returned plain text — fallback to heuristics below
                parsed = {}
            if parsed:
                # normalize parsed fields
                passed = bool(parsed.get("passed", False))
                issues = parsed.get("issues", [])
                recs = parsed.get("recommendations", [])
                fixed = parsed.get("fixed_code", None)
                append_log("audit_agent", "audit", json.dumps({"passed": passed, "issues": issues})[:1000], wfn=campaign.get("WFN",""), action="audit")
                return {"passed": passed, "issues": issues, "recommendations": recs, "fixed_code": fixed}
            # else fallback to heuristics
        except Exception as e:
            append_log("audit_agent", "warning", f"LLM audit failed: {e}", wfn=campaign.get("WFN",""), action="audit_fallback")

        # Heuristic audit fallback
        issues = []
        recs = []
        lower = (sas_text or "").lower()
        if "segmentation_requirements" not in lower and "segmentation" not in lower and "proc sql" not in lower:
            issues.append("Segmentation read/filter step might be missing or not obvious.")
            recs.append("Ensure segmentation file is read and filters applied to target dataset.")
        if "rand(" not in lower and "call streaminit" not in lower and "rand('uniform')" not in lower:
            issues.append("No random assignment detected for test/control.")
            recs.append("Add call streaminit and random assignment (rand('uniform') or similar).")
        if expected_export_name.lower() not in lower:
            issues.append(f"Export dataset name does not match required naming: {expected_export_name}")
            recs.append(f"Ensure export dataset is named exactly: {expected_export_name}")
        # additional heuristic checks can be added
        passed = len(issues) == 0
        append_log("audit_agent", "audit_fallback", json.dumps({"passed": passed, "issues": issues})[:1000], wfn=campaign.get("WFN",""), action="audit_fallback")
        return {"passed": passed, "issues": issues, "recommendations": recs, "fixed_code": None}
    except Exception as e:
        append_log("audit_agent", "error", f"Exception: {e}", wfn=campaign.get("WFN",""), action="error", notes=traceback.format_exc())
        return {"passed": False, "issues": ["Audit exception"], "recommendations": [], "fixed_code": None}

# ---------------------------
# Small orchestrator helper (useful for notebook testing)
# ---------------------------
def orchestrate_once(wfn: str, user_confirm_requirements: bool = True, user_confirm_code: bool = True) -> Dict[str, Any]:
    """
    Orchestrate one flow using the agents: gather -> generate -> audit -> save (if approved).
    This function uses the local LLM wrappers (which may call Bedrock or fallback).
    """
    try:
        camp_res = campaign_requirements_agent(wfn)
        if not camp_res.get("found"):
            append_log("orchestrator", "info", f"No campaign found for {wfn}", wfn=wfn, action="no_campaign")
            return {"status": "no_campaign", "message": camp_res.get("message", "")}
        campaign = camp_res["campaign"]
        segments = camp_res["segments"]
        append_log("orchestrator", "info", "requirements presented", wfn=wfn, action="present_requirements")

        if not user_confirm_requirements:
            append_log("orchestrator", "info", "requirements rejected (simulated)", wfn=wfn, action="requirements_rejected")
            return {"status": "requirements_rejected"}

        append_log("orchestrator", "info", "requirements approved (simulated)", wfn=wfn, action="requirements_approved")

        cg = code_generation_agent(campaign, segments, wfn)
        sas_text = cg.get("sas", "")
        export_name = cg.get("export_name", "")
        append_log("orchestrator", "info", "code generated", wfn=wfn, action="code_generated")

        audit = code_audit_agent(sas_text, campaign, segments)
        iterations = 0
        while (not audit.get("passed", False)) and iterations < 3:
            iterations += 1
            # naive automatic refinement: prepend recommendations as comments then re-audit
            patch = "\n/* Audit refinement attempt: " + "; ".join(audit.get("recommendations", ["no rec"])) + " */\n"
            sas_text = patch + sas_text
            audit = code_audit_agent(sas_text, campaign, segments)
            append_log("orchestrator", "info", f"auto refinement #{iterations}", wfn=wfn, action="auto_refine")
            if audit.get("passed"):
                break

        if not user_confirm_code:
            append_log("orchestrator", "info", "code rejected (simulated)", wfn=wfn, action="code_rejected")
            return {"status": "code_rejected", "sas": sas_text, "audit": audit, "iterations": iterations}

        # final save
        out_path = KB_DIR / f"generated_sas_{wfn}.sas"
        out_path.write_text(sas_text, encoding="utf-8")
        append_log("orchestrator", "info", f"code approved and saved to {out_path}", wfn=wfn, action="code_saved")
        return {"status": "completed", "sas": sas_text, "audit": audit, "iterations": iterations, "path": str(out_path)}
    except Exception as e:
        append_log("orchestrator", "error", f"Exception: {e}", wfn=wfn, action="error", notes=traceback.format_exc())
        return {"status": "error", "message": str(e)}

# ---------------------------
# End of file
# ---------------------------
