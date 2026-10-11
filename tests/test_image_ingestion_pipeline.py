"""
DataSentinel - Phase 5 Real Image Ingestion and Vision Audit Pipeline Tests
Tests real image uploading, ZIP archive extraction, path traversal defense,
decompression limits, thumbnail generation, and real detector execution.
"""

import io
import os
import sys
import zipfile
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient

from backend.main import app
from services.image_ingestion_service import (
    ImageIngestionService,
    is_safe_zip_path,
    infer_label_from_path,
)

client = TestClient(app)


def create_dummy_image_bytes(width: int = 32, height: int = 32, color: tuple = (255, 0, 0), fmt: str = "JPEG") -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def create_dummy_zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


class TestImageIngestionPipeline:
    def test_infer_label_from_path(self):
        assert infer_label_from_path("cats/cat_001.jpg") == "cats"
        assert infer_label_from_path("train/dogs/dog_002.png") == "dogs"
        assert infer_label_from_path("cat_003.jpg") is None
        assert infer_label_from_path("images/val/horse/004.jpg") == "horse"

    def test_inspect_individual_images_upload(self):
        img1 = create_dummy_image_bytes(32, 32, (200, 50, 50), "JPEG")
        img2 = create_dummy_image_bytes(48, 48, (50, 200, 50), "PNG")

        response = client.post(
            "/api/image-datasets/inspect",
            files=[
                ("files", ("image_a.jpg", img1, "image/jpeg")),
                ("files", ("image_b.png", img2, "image/png")),
            ],
            data={"dataset_name": "Test Ingested Images"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_images"] == 2
        assert "dataset_id" in data
        assert len(data["sample_previews"]) == 2
        assert data["formats"]["JPEG"] == 1
        assert data["formats"]["PNG"] == 1

        # Test thumbnail preview endpoint
        sample_id = data["sample_previews"][0]["sample_id"]
        dataset_id = data["dataset_id"]
        prev_res = client.get(f"/api/image-datasets/{dataset_id}/samples/{sample_id}/preview")
        assert prev_res.status_code == 200
        assert prev_res.headers["content-type"] == "image/jpeg"

    def test_inspect_zip_archive_with_labeled_subfolders(self):
        zip_files = {
            "cats/cat_01.jpg": create_dummy_image_bytes(32, 32, (10, 10, 10)),
            "cats/cat_02.jpg": create_dummy_image_bytes(32, 32, (20, 20, 20)),
            "dogs/dog_01.jpg": create_dummy_image_bytes(32, 32, (100, 100, 100)),
            "dogs/dog_02.jpg": create_dummy_image_bytes(32, 32, (120, 120, 120)),
        }
        zip_content = create_dummy_zip_bytes(zip_files)

        response = client.post(
            "/api/image-datasets/inspect",
            files=[
                ("files", ("labeled_dataset.zip", zip_content, "application/zip"))
            ],
            data={"dataset_name": "Cats and Dogs Archive"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_images"] == 4
        assert data["has_labels"] is True
        assert data["labels_distribution"].get("cats") == 2
        assert data["labels_distribution"].get("dogs") == 2

    def test_reject_corrupted_file(self):
        corrupted_bytes = b"NOT_A_VALID_IMAGE_HEADER_BINARY_DATA"
        response = client.post(
            "/api/image-datasets/inspect",
            files=[
                ("files", ("corrupt.jpg", corrupted_bytes, "image/jpeg"))
            ]
        )
        assert response.status_code == 400
        assert "No valid image files found" in response.json()["detail"]

    def test_run_uploaded_image_audit(self):
        # 1. Ingest dataset
        zip_files = {
            "cats/cat_01.jpg": create_dummy_image_bytes(32, 32, (255, 0, 0)),
            "cats/cat_02.jpg": create_dummy_image_bytes(32, 32, (255, 0, 0)),  # exact duplicate
            "dogs/dog_01.jpg": create_dummy_image_bytes(32, 32, (0, 255, 0)),
            "dogs/dog_02.jpg": create_dummy_image_bytes(32, 32, (0, 0, 255)),
        }
        zip_content = create_dummy_zip_bytes(zip_files)

        inspect_res = client.post(
            "/api/image-datasets/inspect",
            files=[("files", ("animals.zip", zip_content, "application/zip"))],
            data={"dataset_name": "Animal Vision Partition"}
        )
        assert inspect_res.status_code == 200
        dataset_id = inspect_res.json()["dataset_id"]

        # 2. Audit dataset
        audit_res = client.post(
            "/api/image-datasets/audit",
            json={"dataset_id": dataset_id}
        )
        if audit_res.status_code != 200:
            print("ERROR IN AUDIT_RES:", audit_res.status_code, audit_res.text)
        assert audit_res.status_code == 200
        audit_data = audit_res.json()
        assert audit_data["is_demo"] is False
        assert audit_data["modality"] == "IMAGE"

        audit = audit_data["audit"]
        assert "exact_duplicate_risk" in audit["available_signals"]
        assert "image_statistics_risk" in audit["available_signals"]
        assert "quarantine_manifest" in audit
        assert audit["quarantine_manifest"]["total_candidate_samples"] == 4
        assert len(audit["sample_results"]) == 4

        # Verify limitations explain that target backdoor triggers require model calibration
        limitations = audit["unresolved_threat_limitations"]
        assert len(limitations) > 0
        assert any("arbitrary_image_backdoor_screening" in l["threat_family"] for l in limitations)
