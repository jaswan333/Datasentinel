from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse
)
from fastapi.staticfiles import (
    StaticFiles
)
from pydantic import BaseModel

from services.datasentinel_bridge import (
    run_tabular_audit,
    read_csv_with_fallback
)
from services.audit_store import (
    register_dataset,
    get_audit
)
from agents.workflow import (
    agent_workflow
)

load_dotenv()

app = FastAPI(
    title="DataSentinel API",
    version="0.5.0",
    description="Agentic Training-Data Trust & Poisoning Audit API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("runtime/datasets").mkdir(
    parents=True,
    exist_ok=True
)

Path("runtime/results").mkdir(
    parents=True,
    exist_ok=True
)

ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls"
}


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
        "project": "DataSentinel",
        "agent_configured": is_groq_configured(),
        "groq_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    }


@app.post("/api/inspect-columns")
def inspect_columns(
    file: UploadFile = File(...)
):
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

        # Check for 2-header style Excel files (e.g. Unnamed: 0 or X1..X23 where row 0 contains feature names)
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
            n_uni = int(df[c].nunique(dropna=True))
            if any(h in c_lower for h in cat_hints):
                suggested_categoricals.append(c)
            elif df[c].dtype == "object":
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
    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only CSV, XLSX and XLS files "
                "are supported."
            )
        )

    dataset_id = uuid.uuid4().hex[:12]

    save_path = (
        Path("runtime/datasets")
        / f"{dataset_id}{extension}"
    )

    with open(
        save_path,
        "wb"
    ) as buffer:
        buffer.write(
            file.file.read()
        )

    categories = None
    if categorical_columns:
        categories = [
            c.strip()
            for c in categorical_columns.split(",")
            if c.strip()
        ]

    try:
        audit_result = run_tabular_audit(
            file_path=str(save_path),
            target_column=target_column.strip() if target_column and target_column.strip() else None,
            categorical_columns=categories
        )
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
        audit_result=audit_result
    )

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "audit": audit_result
    }


@app.get("/api/audit/{dataset_id}")
def audit_summary(
    dataset_id: str
):
    try:
        return get_audit(
            dataset_id
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )


@app.post("/api/agent/ask")
def ask_agent(
    request: AgentQuestion
):
    if not is_groq_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "GROQ_API_KEY is not configured or contains placeholder. "
                "Please configure GROQ_API_KEY in your .env file."
            )
        )

    try:
        result = agent_workflow.invoke({
            "dataset_id": request.dataset_id,
            "question": request.question
        })
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {exc}"
        )

    return {
        "dataset_id": request.dataset_id,
        "question": request.question,
        "answer": result.get(
            "final_report",
            "No report generated."
        ),
        "audit_agent": result.get(
            "audit_summary"
        ),
        "investigation_agent": result.get(
            "investigation"
        )
    }


@app.get("/")
def frontend():
    index_path = Path("frontend/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found.")
    return FileResponse(
        str(index_path)
    )
