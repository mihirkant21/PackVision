import logging
import threading
import traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
from sqlalchemy.orm import Session

from backend.app.db.session import SessionLocal
from backend.app.db.models import Generation, GenerationOutput, Asset
from backend.app.services.asset_service import AssetService
from backend.app.services.inference_service import inference_service

logger = logging.getLogger(__name__)

# Canonical job lifecycle states.
ACTIVE_STATES = {"queued", "preprocessing", "generating", "postprocessing"}
TERMINAL_STATES = {"completed", "failed", "cancelled"}


class RetryLimitExceeded(Exception):
    """Raised when a generation has already been retried the maximum number of times."""


class JobService:
    """Manages asynchronous background rendering tasks and lifecycle transitions."""

    MAX_RETRIES = 3  # A job may be retried at most 3 times (4th attempt -> 409).

    def __init__(self, max_workers: int = 3):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="packvision-job"
        )
        # Cooperative cancellation registry: job_id -> threading.Event.
        # Workers poll the event at stage boundaries; cancel_job() sets it.
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._flags_lock = threading.Lock()
        # Approximate process start time; used to detect orphaned jobs left
        # behind by a previous process (crash / restart) during recovery.
        self.process_start = datetime.now(timezone.utc)
        logger.info(f"JobService initialized with {max_workers} background workers.")

    # ------------------------------------------------------------------ #
    # Cancellation support
    # ------------------------------------------------------------------ #
    def submit_generation_job(self, generation_id: str) -> None:
        """Enqueue generation task for background execution."""
        logger.info(f"Submitting generation job {generation_id} to background queue.")
        with self._flags_lock:
            # Register the cancel flag before the worker starts so a cancel
            # request that arrives while the job is still queued is honored
            # at the worker's first checkpoint.
            self._cancel_flags.setdefault(generation_id, threading.Event())
        self.executor.submit(self._run_job_lifecycle, generation_id)

    def _cancelled(self, generation_id: str) -> bool:
        """Return True if a cancel was requested for this job."""
        with self._flags_lock:
            event = self._cancel_flags.get(generation_id)
        return bool(event and event.is_set())

    def cancel_job(self, generation_id: str) -> bool:
        """Request cooperative cancellation of a job.

        Sets the cancel flag so the worker stops at its next stage boundary
        and marks the job "cancelled". Returns True if the job was in a
        cancellable (non-terminal) state, False if it was unknown, already
        completed/failed/cancelled, or could not be looked up.
        """
        try:
            with SessionLocal() as db:
                gen = (
                    db.query(Generation)
                    .filter(Generation.id == generation_id)
                    .first()
                )
                if not gen or gen.status in TERMINAL_STATES:
                    return False
        except Exception:
            logger.exception(
                f"[Job {generation_id}] Failed to look up job for cancellation."
            )
            return False

        with self._flags_lock:
            self._cancel_flags.setdefault(
                generation_id, threading.Event()
            ).set()
        logger.info(f"[Job {generation_id}] Cancellation requested.")
        return True

    def _mark_cancelled(self, db: Session, generation_id: str) -> None:
        """Transition a job into the terminal 'cancelled' state."""
        self._update_job_status(
            db=db,
            generation_id=generation_id,
            status="cancelled",
            progress=100,
            stage_message="Generation cancelled by user.",
            completed=True,
            error_code="cancelled",
            error_message="Job cancelled before completion.",
        )

    def _clear_cancel_flag(self, generation_id: str) -> None:
        with self._flags_lock:
            self._cancel_flags.pop(generation_id, None)

    # ------------------------------------------------------------------ #
    # Startup recovery for orphaned jobs
    # ------------------------------------------------------------------ #
    def recover_stuck_jobs(self) -> int:
        """Mark jobs interrupted by a previous process run as failed.

        Any job left in an active state (queued/preprocessing/generating/
        postprocessing) whose created_at predates this process start cannot
        have a live worker anymore (workers are in-process), so it is marked
        failed with error_code "interrupted" and a user-readable message.
        Jobs created during the current process run are left untouched.

        Returns the number of recovered jobs.
        """
        recovered = 0
        try:
            with SessionLocal() as db:
                stuck = (
                    db.query(Generation)
                    .filter(Generation.status.in_(sorted(ACTIVE_STATES)))
                    .all()
                )
                for gen in stuck:
                    created = gen.created_at
                    if created is None:
                        continue
                    if created.tzinfo is None:
                        # SQLite returns naive datetimes; they are stored as UTC.
                        created = created.replace(tzinfo=timezone.utc)
                    if created >= self.process_start:
                        # Created by the current process; a worker may own it.
                        continue
                    gen.status = "failed"
                    gen.error_code = "interrupted"
                    gen.error_message = (
                        "Server restarted during processing — please retry"
                    )
                    gen.stage_message = "Interrupted by server restart"
                    gen.progress = 100
                    gen.completed_at = datetime.now(timezone.utc)
                    recovered += 1
                if recovered:
                    db.commit()
                    logger.info(
                        f"Startup recovery: marked {recovered} orphaned job(s) as failed."
                    )
        except Exception:
            logger.exception("Startup recovery of stuck jobs failed.")
        return recovered

    # ------------------------------------------------------------------ #
    # State machine
    # ------------------------------------------------------------------ #
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
            # 0. Cancellation checkpoint: cancelled while queued.
            if self._cancelled(generation_id):
                self._mark_cancelled(db, generation_id)
                logger.info(f"[Job {generation_id}] Cancelled before processing.")
                return

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
            input_mode = str(params.get("input_mode") or "auto")

            # Cancellation checkpoint: cancelled during preprocessing.
            if self._cancelled(generation_id):
                self._mark_cancelled(db, generation_id)
                logger.info(f"[Job {generation_id}] Cancelled before generation stage.")
                return

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
            # NOTE: cancellation is only honored at stage boundaries because
            # inference_service.generate() does not accept a cancel callback.
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
                input_mode=input_mode,
                progress_callback=on_progress,
            )

            # Cancellation checkpoint: cancelled during generation. Results
            # are discarded so a cancelled job never publishes outputs.
            if self._cancelled(generation_id):
                self._mark_cancelled(db, generation_id)
                logger.info(f"[Job {generation_id}] Cancelled after generation stage.")
                return

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

                # Record the engine that actually rendered this job (honest
                # model provenance instead of the creation-time SDXL default).
                engine_used = metrics.get("engine_used")
                if engine_used and idx == 0:
                    gen.model_name = str(engine_used)
                    engine_note = metrics.get("engine_note")
                    if engine_note:
                        gen.model_revision = str(engine_note)[:100]

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
            self._clear_cancel_flag(generation_id)

    # ------------------------------------------------------------------ #
    # Retry with cap
    # ------------------------------------------------------------------ #
    def retry_generation(self, db: Session, generation_id: str) -> Optional[Generation]:
        """Reset a failed generation and re-submit it to the queue.

        Enforces a maximum of :data:`MAX_RETRIES` retries per job. The retry
        count is persisted in the existing ``parameters_json`` JSON column
        (under the "retry_count" key) so it survives process restarts without
        a schema migration. Raises :class:`RetryLimitExceeded` when the cap
        is reached.
        """
        gen = db.query(Generation).filter(Generation.id == generation_id).first()
        if not gen:
            return None

        params = dict(gen.parameters_json or {})
        attempts = int(params.get("retry_count", 0) or 0)
        if attempts >= self.MAX_RETRIES:
            raise RetryLimitExceeded(
                f"Generation '{generation_id}' has reached the maximum of "
                f"{self.MAX_RETRIES} retries and cannot be retried again."
            )
        params["retry_count"] = attempts + 1
        # Reassign (not mutate) so SQLAlchemy detects the JSON column change.
        gen.parameters_json = params

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

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #
    def cancel_active_jobs(self) -> int:
        """Signal every registered in-flight/queued job to cancel.

        Only sets the cooperative cancel events — workers stop at their next
        stage boundary and mark their job 'cancelled' (including the
        checkpoint right after the generation stage, so results produced by a
        worker during teardown are discarded). Never touches the executor.
        """
        with self._flags_lock:
            events = list(self._cancel_flags.values())
        for event in events:
            event.set()
        return len(events)

    def shutdown(self, wait: bool = False, cancel_running: bool = True) -> None:
        """Shut down the worker pool, optionally cancelling in-flight jobs.

        WARNING: this permanently disables this JobService instance's
        executor — any later submit() raises RuntimeError. Callers must only
        do this at final process exit (never between test cases; see the
        PACKVISION_TESTING guard in main.py lifespan teardown).
        """
        if cancel_running:
            self.cancel_active_jobs()
        self.executor.shutdown(wait=wait, cancel_futures=not wait)


job_service = JobService()
