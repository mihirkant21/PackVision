import logging
import traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from sqlalchemy.orm import Session

from backend.app.db.session import SessionLocal
from backend.app.db.models import Generation, GenerationOutput, Asset
from backend.app.services.asset_service import AssetService
from backend.app.services.inference_service import inference_service

logger = logging.getLogger(__name__)


class JobService:
    """Manages asynchronous background rendering tasks and lifecycle transitions."""

    def __init__(self, max_workers: int = 3):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="packvision-job"
        )
        logger.info(f"JobService initialized with {max_workers} background workers.")

    def submit_generation_job(self, generation_id: str) -> None:
        """Enqueue generation task for background execution."""
        logger.info(f"Submitting generation job {generation_id} to background queue.")
        self.executor.submit(self._run_job_lifecycle, generation_id)

    def _update_job_status(
        self,
        db: Session,
        generation_id: str,
        status: str,
        progress: int,
        stage_message: str,
        started: bool = False,
        completed: bool = False,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Generation]:
        """Update job progress and state atomically."""
        gen = db.query(Generation).filter(Generation.id == generation_id).first()
        if not gen:
            return None

        # If user cancelled, do not override
        if gen.status == "cancelled":
            return gen

        gen.status = status
        gen.progress = progress
        gen.stage_message = stage_message

        if started and not gen.started_at:
            gen.started_at = datetime.now(timezone.utc)

        if completed:
            gen.completed_at = datetime.now(timezone.utc)

        if error_code:
            gen.error_code = error_code
        if error_message:
            gen.error_message = error_message

        db.commit()
        db.refresh(gen)
        return gen

    def _run_job_lifecycle(self, generation_id: str) -> None:
        """Full state-machine execution running in background thread."""
        logger.info(f"[Job {generation_id}] Starting generation execution.")
        db: Session = SessionLocal()

        try:
            # 1. State: Queued -> Preprocessing
            gen = self._update_job_status(
                db=db,
                generation_id=generation_id,
                status="preprocessing",
                progress=10,
                stage_message="Validating assets and preparing canvas...",
                started=True,
            )
            if not gen or gen.status == "cancelled":
                logger.info(f"[Job {generation_id}] Terminated early (cancelled or deleted).")
                return

            # Retrieve input assets
            input_asset = db.query(Asset).filter(Asset.id == gen.input_asset_id).first()
            if not input_asset:
                raise ValueError(f"Input asset '{gen.input_asset_id}' not found.")

            input_sketch_path = AssetService.get_asset_file_path(input_asset)
            if not input_sketch_path.exists():
                raise FileNotFoundError(
                    f"Input sketch file does not exist at {input_sketch_path}"
                )

            reference_path = None
            if gen.reference_asset_id:
                ref_asset = (
                    db.query(Asset).filter(Asset.id == gen.reference_asset_id).first()
                )
                if ref_asset:
                    reference_path = AssetService.get_asset_file_path(ref_asset)

            logo_path = None
            if gen.logo_asset_id:
                logo_asset = db.query(Asset).filter(Asset.id == gen.logo_asset_id).first()
                if logo_asset:
                    logo_path = AssetService.get_asset_file_path(logo_asset)

            # Extract generation parameters
            params = gen.parameters_json or {}
            num_outputs = int(params.get("num_outputs", 1))
            guidance_scale = float(params.get("guidance_scale", 7.5))
            controlnet_scale = float(params.get("controlnet_scale", 0.85))
            steps = int(params.get("steps", 30))
            logo_placement = params.get("logo_placement")

            # 2. State: Preprocessing -> Generating
            self._update_job_status(
                db=db,
                generation_id=generation_id,
                status="generating",
                progress=30,
                stage_message="Synthesizing photorealistic packaging render...",
            )

            # Progress callback bridge
            def on_progress(pct: int, msg: str):
                try:
                    with SessionLocal() as progress_db:
                        self._update_job_status(
                            db=progress_db,
                            generation_id=generation_id,
                            status="generating",
                            progress=pct,
                            stage_message=msg,
                        )
                except Exception as ex:
                    logger.warning(f"Failed to update progress: {ex}")

            # Run AI Inference
            rendered_results = inference_service.generate(
                input_sketch_path=input_sketch_path,
                package_type=gen.package_type,
                material=gen.material,
                finish=gen.finish,
                viewpoint=gen.viewpoint,
                background=gen.background,
                user_prompt=gen.prompt,
                negative_prompt=gen.negative_prompt,
                reference_image_path=reference_path,
                logo_image_path=logo_path,
                logo_placement=logo_placement,
                num_outputs=num_outputs,
                seed=gen.seed,
                guidance_scale=guidance_scale,
                controlnet_scale=controlnet_scale,
                steps=steps,
                progress_callback=on_progress,
            )

            # 3. State: Generating -> Postprocessing
            self._update_job_status(
                db=db,
                generation_id=generation_id,
                status="postprocessing",
                progress=85,
                stage_message="Saving output renders and generating client thumbnails...",
            )

            # Save generated images as Assets and link GenerationOutput
            for idx, res in enumerate(rendered_results):
                img = res["image"]
                metrics = res.get("metrics", {})

                label = f"{gen.package_type}_{gen.material or 'render'}_{idx + 1}.png"
                output_asset = AssetService.save_output_image(
                    db=db,
                    image=img,
                    file_name_label=label,
                    user_id=gen.user_id,
                )

                gen_output = GenerationOutput(
                    generation_id=gen.id,
                    asset_id=output_asset.id,
                    is_selected=(idx == 0),
                    metrics_json=metrics,
                )
                db.add(gen_output)

            db.commit()

            # 4. State: Completed
            self._update_job_status(
                db=db,
                generation_id=generation_id,
                status="completed",
                progress=100,
                stage_message="Packaging renders generated successfully.",
                completed=True,
            )
            logger.info(f"[Job {generation_id}] Finished successfully.")

        except Exception as e:
            logger.error(
                f"[Job {generation_id}] Execution failed: {str(e)}\n{traceback.format_exc()}"
            )
            self._update_job_status(
                db=db,
                generation_id=generation_id,
                status="failed",
                progress=100,
                stage_message=f"Generation failed: {str(e)}",
                completed=True,
                error_code="INFERENCE_FAILED",
                error_message=str(e),
            )
        finally:
            db.close()

    def retry_generation(self, db: Session, generation_id: str) -> Optional[Generation]:
        """Reset a failed generation and re-submit it to the queue."""
        gen = db.query(Generation).filter(Generation.id == generation_id).first()
        if not gen:
            return None

        # Reset outputs
        db.query(GenerationOutput).filter(
            GenerationOutput.generation_id == generation_id
        ).delete()

        gen.status = "queued"
        gen.progress = 0
        gen.stage_message = "Generation job re-queued"
        gen.error_code = None
        gen.error_message = None
        gen.started_at = None
        gen.completed_at = None
        db.commit()
        db.refresh(gen)

        self.submit_generation_job(gen.id)
        return gen


job_service = JobService()
