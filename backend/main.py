from __future__ import annotations

import os
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    Body
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse
)
from pydantic import BaseModel

from services.datasentinel_bridge import (
    run_tabular_audit,
    read_csv_with_fallback
)
from services.audit_store import (
    register_dataset,
    get_audit,
    get_dataset,
    list_audit_history,
)
from core.detector_adapters import (
    adapt_tabular_audit_to_unified,
    adapt_image_audit_to_unified
)
from agents.workflow import (
    agent_workflow
)

load_dotenv()

app = FastAPI(
    title="DataSentinel API",
    version="0.8.0",
    description="Agentic Multi-Modal Dataset Trust & Poisoning Defense API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("runtime/datasets").mkdir(parents=True, exist_ok=True)
Path("runtime/results").mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class AgentQuestion(BaseModel):
    dataset_id: str
    question: str


def is_groq_configured() -> bool:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(key and not key.startswith("your_") and key != "gsk_placeholder_key")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "DataSentinel Backend API",
        "version": "0.8.0",
        "policy_version": "1.0.0-evidence-weighted",
        "project": "DataSentinel",
        "milestone": "M8 - Platform Integration & Unified Auditing",
        "agent_configured": is_groq_configured(),
        "groq_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    }


@app.post("/api/inspect-columns")
def inspect_columns(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only CSV, XLSX and XLS files are supported."
        )

    try:
        content = file.file.read()
        import io
        import pandas as pd
        if extension == ".csv":
            df = read_csv_with_fallback(io.BytesIO(content), nrows=100)
        else:
            df = pd.read_excel(io.BytesIO(content), nrows=100)

        # Check for 2-header style Excel files
        if len(df) > 1:
            first_row = df.iloc[0]
            col_names = [str(c).strip() for c in df.columns]
            if any("unnamed" in c.lower() for c in col_names) or any(c.lower().startswith("x") for c in col_names):
                if all(isinstance(v, str) and not v.strip().replace(".", "").isdigit() for v in first_row if pd.notna(v)):
                    new_cols = [str(v).strip() for v in first_row]
                    df = df.iloc[1:].reset_index(drop=True)
                    df.columns = new_cols

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        cols = [str(c) for c in df.columns]
        exact_target_hints = {"target", "label", "class", "y", "default", "churn", "fraud", "is_fraud"}
        substring_target_hints = ["default_payment", "next_month", "target_col", "target_var"]

        suggested_target = None
        for c in cols:
            norm_c = c.lower().strip().replace(" ", "_").replace(".", "_")
            if norm_c in exact_target_hints or any(sub in norm_c for sub in substring_target_hints):
                suggested_target = c
                break

        cat_hints = ["sex", "education", "marriage", "gender", "status", "category", "type"]
        suggested_categoricals = []
        for c in cols:
            if c == suggested_target or c.lower().strip() in {"id", "index", "unnamed: 0", "is_poisoned", "poison_label", "poisoned", "ground_truth"}:
                continue
            c_lower = c.lower().strip()
            if any(h in c_lower for h in cat_hints) or df[c].dtype == "object":
                suggested_categoricals.append(c)

        column_stats = []
        for c in cols:
            column_stats.append({
                "name": c,
                "dtype": str(df[c].dtype),
                "unique_count": int(df[c].nunique(dropna=True)),
                "is_suggested_cat": c in suggested_categoricals,
                "is_suggested_target": c == suggested_target
            })

        return {
            "filename": file.filename,
            "columns": cols,
            "suggested_target": suggested_target,
            "suggested_categoricals": suggested_categoricals,
            "column_stats": column_stats
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to inspect columns: {str(exc)}"
        )


@app.post("/api/audit")
def audit_dataset(
    file: UploadFile = File(...),
    target_column: str | None = Form(None),
    categorical_columns: str | None = Form(None)
):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only CSV, XLSX and XLS files are supported."
        )

    dataset_id = uuid.uuid4().hex[:12]
    save_path = Path("runtime/datasets") / f"{dataset_id}{extension}"

    with open(save_path, "wb") as buffer:
        buffer.write(file.file.read())

    categories = None
    if categorical_columns:
        categories = [c.strip() for c in categorical_columns.split(",") if c.strip()]

    try:
        raw_audit = run_tabular_audit(
            file_path=str(save_path),
            target_column=target_column.strip() if target_column and target_column.strip() else None,
            categorical_columns=categories
        )

        unified = adapt_tabular_audit_to_unified(
            raw_audit,
            dataset_name=file.filename or f"dataset_{dataset_id}.csv",
            file_path=str(save_path)
        )
        audit_payload = unified.to_legacy_dict()

    except Exception as exc:
        try:
            save_path.unlink()
        except OSError:
            pass

        raise HTTPException(
            status_code=500,
            detail=f"Audit execution error: {str(exc)}"
        )

    register_dataset(
        dataset_id=dataset_id,
        file_path=str(save_path),
        audit_result=audit_payload
    )

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "audit": audit_payload
    }


@app.post("/api/audit/benchmark/{benchmark_id}")
def load_benchmark_audit(benchmark_id: str):
    """
    Instantly loads pre-computed benchmark audits (Tabular or Image DL)
    into the unified store for live dashboard exploration and agentic review.
    """
    root = Path(__file__).resolve().parents[1]
    bid = benchmark_id.lower().strip()

    if bid in {"image_backdoor", "cifar10_backdoor", "backdoor_patch"}:
        summary_path = root / "experiments/image/backdoor_patch/attack_summary.json"
        evidence_path = root / "experiments/image/backdoor_patch/sample_evidence.csv"
        ds_id = "m6_cifar10_backdoor"
        ds_name = "CIFAR-10 3% Trojan Backdoor (Patch Trigger)"
        modality = "IMAGE"

        if not summary_path.exists():
            raise HTTPException(status_code=404, detail=f"Benchmark summary not found: {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as f:
            raw_audit = json.load(f)

        unified = adapt_image_audit_to_unified(
            raw_audit,
            dataset_name=ds_name,
            evidence_csv_path=str(evidence_path) if evidence_path.exists() else None
        )

    elif bid in {"image_clean_label", "cifar10_clean_label", "clean_label"}:
        summary_path = root / "experiments/image/clean_label/attack_summary.json"
        evidence_path = root / "experiments/image/clean_label/sample_evidence.csv"
        ds_id = "m6_cifar10_clean_label"
        ds_name = "CIFAR-10 5% Clean-Label Manipulation (Centroid Blend)"
        modality = "IMAGE"

        if not summary_path.exists():
            raise HTTPException(status_code=404, detail=f"Benchmark summary not found: {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as f:
            raw_audit = json.load(f)

        unified = adapt_image_audit_to_unified(
            raw_audit,
            dataset_name=ds_name,
            evidence_csv_path=str(evidence_path) if evidence_path.exists() else None
        )

    elif bid in {"tabular_credit", "tabular_credit_sample", "credit_card"}:
        sample_path = root / "poisoned_credit_sample_1000.csv"
        ds_id = "m4_credit_sample_1000"
        ds_name = "UCI Credit Default (5% Label Flip Benchmark)"
        modality = "TABULAR"

        if not sample_path.exists():
            raise HTTPException(status_code=404, detail="poisoned_credit_sample_1000.csv not found")

        raw_audit = run_tabular_audit(
            file_path=str(sample_path),
            target_column="default_payment_next_month"
        )
        unified = adapt_tabular_audit_to_unified(
            raw_audit,
            dataset_name=ds_name,
            file_path=str(sample_path)
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown benchmark ID: {benchmark_id}. Supported: 'image_backdoor', 'image_clean_label', 'tabular_credit'"
        )

    payload = unified.to_legacy_dict()
    payload["dataset_id"] = ds_id
    register_dataset(dataset_id=ds_id, file_path=f"benchmark/{bid}", audit_result=payload)

    return {
        "dataset_id": ds_id,
        "filename": ds_name,
        "modality": modality,
        "audit": payload
    }


@app.get("/api/audit/history")
def get_audit_history(limit: int = 15):
    """
    Returns real persisted audit history across tabular and vision modalities.
    """
    return list_audit_history(limit=limit)


@app.get("/api/audit/{dataset_id}")
def audit_summary(dataset_id: str):
    try:
        return get_audit(dataset_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_id}' not found in audit store."
        )


@app.get("/api/audit/{dataset_id}/quarantine")
def get_quarantine_details(dataset_id: str):
    try:
        audit = get_audit(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    manifest = audit.get("quarantine_manifest") or {}
    samples = audit.get("sample_results", [])
    high_risk_samples = [s for s in samples if s.get("risk_tier") == "HIGH" or s.get("risk_level") == "HIGH"]

    return {
        "dataset_id": dataset_id,
        "manifest": manifest,
        "confirmation_status": manifest.get("confirmation_status", "PENDING_OPERATOR_REVIEW"),
        "quarantined_count": manifest.get("quarantined_count", audit.get("high_risk_quarantine", 0)),
        "operating_threshold": manifest.get("operating_threshold"),
        "threshold_policy": manifest.get("threshold_policy"),
        "operator_notes": manifest.get("operator_notes"),
        "quarantined_samples": high_risk_samples[:100]
    }


@app.post("/api/audit/{dataset_id}/quarantine/confirm")
def confirm_quarantine(dataset_id: str, request_data: dict = Body(default={})):
    """
    Operator sign-off endpoint confirming that quarantined records have been reviewed
    and approved for non-destructive screening before model retraining.
    """
    try:
        audit = get_audit(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    manifest = audit.get("quarantine_manifest", {})
    notes = request_data.get("notes", "Operator sign-off verified via DataSentinel")
    manifest["confirmation_status"] = "CONFIRMED"
    manifest["human_review_confirmed"] = True
    manifest["operator_notes"] = notes
    manifest["confirmed_at"] = str(Path().stat().st_mtime)
    audit["quarantine_manifest"] = manifest
    audit["status"] = "QUARANTINE_CONFIRMED"
    audit["recommended_action"] = "Quarantine verified by human operator. Proceed with screened retraining."

    register_dataset(dataset_id=dataset_id, file_path=audit.get("file_path", ""), audit_result=audit)

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "quarantined_samples": manifest.get("quarantined_count", audit.get("high_risk_quarantine", 0)),
        "quarantined_count": manifest.get("quarantined_count", audit.get("high_risk_quarantine", 0)),
        "quarantine_status": "CONFIRMED",
        "operator_notes": notes,
        "message": "Quarantine manifest confirmed by human operator. Non-destructive screening authorized."
    }


@app.get("/api/audit/{dataset_id}/limitations")
def get_threat_limitations(dataset_id: str):
    try:
        audit = get_audit(dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    limitations = audit.get("unresolved_threat_limitations", [])
    return {
        "dataset_id": dataset_id,
        "unresolved_threat_limitations": limitations
    }


@app.post("/api/agent/ask")
def ask_agent(request: AgentQuestion):
    try:
        result = agent_workflow.invoke({
            "dataset_id": request.dataset_id,
            "question": request.question
        })
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent workflow error: {exc}"
        )

    return {
        "dataset_id": request.dataset_id,
        "question": request.question,
        "answer": result.get("final_report", "No report generated."),
        "audit_agent": result.get("audit_summary"),
        "investigation_agent": result.get("investigation")
    }


# =====================================================================
# REAL IMAGE DATASET INGESTION & VISION SECURITY AUDIT (PHASE 5)
# =====================================================================

from services.image_ingestion_service import ImageIngestionService


class ImageAuditRequest(BaseModel):
    dataset_id: str


@app.post("/api/image-datasets/inspect")
async def inspect_image_dataset(
    files: List[UploadFile] = File(...),
    dataset_name: Optional[str] = Form(None)
):
    """
    Ingests and validates uploaded image files (JPG, JPEG, PNG, WEBP) or ZIP archive.
    Performs decompression bomb and path traversal checks, decodes dimensions, and builds thumbnails.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    extracted_files: List[Tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        extracted_files.append((f.filename or "uploaded_image", content))

    try:
        inspection = ImageIngestionService.inspect_and_stage_images(
            files=extracted_files,
            dataset_name=dataset_name
        )
        return inspection
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image ingestion error: {exc}")


@app.get("/api/image-datasets/{dataset_id}")
def get_image_dataset_inspection(dataset_id: str):
    try:
        meta = ImageIngestionService.get_staged_dataset(dataset_id)
        return meta["inspection"]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Staged image dataset '{dataset_id}' not found.")


@app.get("/api/image-datasets/{dataset_id}/samples")
def get_image_dataset_samples(
    dataset_id: str,
    page: int = 1,
    page_size: int = 24
):
    try:
        meta = ImageIngestionService.get_staged_dataset(dataset_id)
        samples = meta["samples"]
        start = (page - 1) * page_size
        end = start + page_size
        paged_samples = samples[start:end]

        return {
            "dataset_id": dataset_id,
            "total_samples": len(samples),
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (len(samples) + page_size - 1) // page_size),
            "samples": [
                {
                    **s,
                    "preview_url": f"/api/image-datasets/{dataset_id}/samples/{s['sample_id']}/preview"
                }
                for s in paged_samples
            ]
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Staged image dataset '{dataset_id}' not found.")


@app.get("/api/image-datasets/{dataset_id}/samples/{sample_id}/preview")
def get_image_sample_preview(dataset_id: str, sample_id: str):
    try:
        thumb_path = ImageIngestionService.get_thumbnail_path(dataset_id, sample_id)
        return FileResponse(str(thumb_path), media_type="image/jpeg")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Sample preview '{sample_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preview error: {exc}")


@app.post("/api/image-datasets/audit")
def run_uploaded_image_audit(request: ImageAuditRequest):
    """
    Executes vision security detectors (Group C duplicates & stats, Group B deep embeddings)
    on real uploaded image datasets without manufacturing benchmark scores.
    """
    try:
        unified_result = ImageIngestionService.run_image_security_audit(request.dataset_id)
        return {
            "dataset_id": request.dataset_id,
            "filename": unified_result.dataset_name,
            "modality": "IMAGE",
            "is_demo": False,
            "audit": unified_result.model_dump()
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image security audit error: {exc}")


@app.get("/")
def frontend():
    index_path = Path("frontend/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found.")
    return FileResponse(str(index_path))
