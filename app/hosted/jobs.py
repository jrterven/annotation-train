"""Durable admission, fair-queue metadata, quota settlement and user cancellation."""
from datetime import date
import shutil
from fastapi import HTTPException
from sqlalchemy import func, select
from .models import InferenceUsage, Job, Project, ImageObject, utcnow, identity

TERMINAL = {"succeeded", "failed", "cancelled"}


def finalize_quota(session, job, charged):
    """Called in the same transaction as terminal state; safe on repeated settlement."""
    job.payload = {}  # Visual references and prompts are unnecessary after execution.
    if not job.quota_reserved:
        return
    usage = session.execute(select(InferenceUsage).where(InferenceUsage.user_id == job.user_id,
        InferenceUsage.day == job.quota_day).with_for_update()).scalar_one()
    usage.reserved -= 1
    if charged:
        usage.used += 1
    job.quota_reserved = False


def enqueue(db, settings, user_id, project_id, image_id, kind, payload):
    if settings.min_free_disk_bytes and shutil.disk_usage(settings.data_dir).free < settings.min_free_disk_bytes:
        raise HTTPException(503, "Server storage is temporarily unavailable")
    day = utcnow().date().isoformat()
    with db.session() as session:
        db.global_lock(session)  # Serializes admission and creation of the per-day counter.
        project = session.get(Project, project_id)
        if not project or project.user_id != user_id or project.deleted_at:
            raise HTTPException(404, "Project not found")
        image = session.scalar(select(ImageObject.id).where(ImageObject.project_id == project_id,
            ImageObject.image_id == image_id, ImageObject.status == "ready"))
        if not image:
            raise HTTPException(404, "Image not found")
        pending = session.scalar(select(func.count()).select_from(Job).where(Job.status == "queued"))
        own_pending = session.scalar(select(func.count()).select_from(Job).where(Job.status == "queued", Job.user_id == user_id))
        if pending >= settings.max_pending_global or own_pending >= settings.max_pending_user:
            raise HTTPException(429, "Inference queue is full; wait for a pending job to finish")
        usage = session.get(InferenceUsage, (user_id, day))
        if not usage:
            usage = InferenceUsage(user_id=user_id, day=day, used=0, reserved=0)
            session.add(usage)
        if usage.used + usage.reserved >= settings.inference_limit:
            raise HTTPException(429, "Daily inference quota exceeded; resets at 00:00 UTC")
        usage.reserved += 1
        job = Job(id=identity(), user_id=user_id, project_id=project_id, image_id=image_id,
                  kind=kind, payload=payload, status="queued", quota_day=day, quota_reserved=True)
        session.add(job)
        session.flush()
        return serialize(job)


def serialize(job):
    result = {"id": job.id, "status": job.status, "cancel_requested": job.cancel_requested,
              "created_at": job.created_at.isoformat() + "Z"}
    if job.status == "succeeded":
        result["result"] = job.result
    if job.status == "failed":
        result["error"] = job.error or "Inference failed"
    return result


def cancel(db, user_id, job_id):
    with db.session() as session:
        db.global_lock(session)
        job = session.execute(select(Job).where(Job.id == job_id, Job.user_id == user_id).with_for_update()).scalar_one_or_none()
        if job is None:
            raise HTTPException(404, "Job not found")
        if job.status == "queued":
            job.cancel_requested = True
            job.status = "cancelled"
            job.finished_at = utcnow()
            finalize_quota(session, job, charged=False)
        elif job.status == "running":
            job.cancel_requested = True
        return serialize(job)
