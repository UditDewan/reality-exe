"""FastAPI service. The browser streams camera frames (or drives the synthetic tabletop); the OpenCV agent
runs here on AWS and streams annotated frames, decisions and traces back. Traces go to S3 if TRACE_BUCKET is set.

    uvicorn api:app --host 0.0.0.0 --port 8000
"""
import asyncio
import base64
import json
import logging
import math
import os
import struct
import time
import uuid

import cv2 as cv
import numpy as np
from fastapi import FastAPI, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import identify
import planner
from controller import Controller
from evaluate import layout
from scene import Obj, Scene, load_config
from vision import Vision
from world import World, marker_sheet

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("reality")
CFG = load_config()
BUCKET = os.environ.get("TRACE_BUCKET")
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "8"))
MAX_FRAME = 2_000_000
COMMANDS = {"place", "release", "reset_ball", "move_obstacle", "occlude", "auto", "new_layout", "restart", "set"}
PIECES = {"guide", "target", "obstacle"}
sessions = {}
app = FastAPI(title="REALITY.EXE")


def nums(m, *keys):
    """Finite floats for `keys` of a browser message, or None if any is missing or malformed."""
    try:
        v = [float(m[k]) for k in keys]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return v if all(map(math.isfinite, v)) else None


def on_board(x, y, a):
    """A dragged pose kept clear of the corner markers and the release gate; angle folded to [-90, 90)."""
    return min(max(x, 60.0), CFG.board_w - 60), min(max(y, 90.0), CFG.board_h - 50), (a + 90) % 180 - 90


def save_trace(sid, ctrl, frame=None):
    """Trial evidence: the full decision trace locally, plus S3 (trace JSON + annotated frame) when configured."""
    doc = json.dumps({"session": sid, "params": ctrl._params(), "trials": ctrl.trials, "trace": ctrl.trace}, default=float)
    os.makedirs("traces", exist_ok=True)
    with open(f"traces/{sid}.json", "w") as f:
        f.write(doc)
    if not BUCKET:
        return
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.put_object(Bucket=BUCKET, Key=f"traces/{sid}.json", Body=doc, ContentType="application/json")
        if frame is not None:
            s3.put_object(Bucket=BUCKET, Key=f"evidence/{sid}/trial-{len(ctrl.trials)}.jpg",
                          Body=cv.imencode(".jpg", frame)[1].tobytes(), ContentType="image/jpeg")
    except Exception as e:  # evidence upload must never kill a live session
        log.warning(json.dumps({"session": sid, "event": "s3_upload_failed", "error": str(e)}))


def packet(ctrl, img, extra=None):
    ok, jpg = cv.imencode(".jpg", ctrl.annotate(img), [cv.IMWRITE_JPEG_QUALITY, 75])
    return json.dumps({"type": "frame", "img": "data:image/jpeg;base64," + base64.b64encode(jpg).decode(),
                       "status": ctrl.status(), **(extra or {})}, default=float)


@app.get("/health")
def health():
    import newton
    return {"ok": True, "opencv": cv.__version__, "newton": newton.__version__, "sessions": len(sessions),
            "trace_bucket": bool(BUCKET)}


@app.post("/api/observe")
async def observe(image: UploadFile):
    """Stateless OpenCV check: upload a photo of the board, get the measured scene back (board mm)."""
    data = await image.read()
    if len(data) > MAX_FRAME:
        raise HTTPException(413, "image too large")
    img = cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not an image")
    return Vision(CFG).observe(img, 0.0).summary()


class PlanJob(BaseModel):
    target: tuple[float, float]
    obstacles: list[tuple[float, float, float, float, float]] = []  # x, y, angle_deg, length, width (mm)
    guide: tuple[float, float, float] | None = None
    params: tuple[float, float, float, float] | None = None


@app.post("/api/plan")
def plan_job(job: PlanJob):
    """Stateless planning job on a supplied layout (no camera needed)."""
    sc = Scene(target=Obj("target", "target", *job.target),
               obstacles=[Obj(f"obstacle-{i + 1}", "obstacle", *o) for i, o in enumerate(job.obstacles)],
               guide=Obj("guide", "guide", *job.guide, CFG.guide_len, CFG.guide_w) if job.guide else None)
    samples = identify.prior_samples(np.array(job.params)) if job.params else identify.prior_samples()
    p = planner.plan(CFG, sc, samples, current=sc.guide)
    p["instruction"] = planner.instruction(sc.guide, p["guide"])
    return p


@app.get("/markers.png")
def markers():
    """Printable ArUco sheet for the physical board (print at 100%)."""
    return Response(cv.imencode(".png", marker_sheet(CFG))[1].tobytes(), media_type="image/png",
                    headers={"Content-Disposition": 'attachment; filename="reality-exe-markers.png"'})


@app.get("/api/sessions/{sid}/trace")
def trace(sid: str):
    if sid in sessions:
        return sessions[sid].trace
    path = f"traces/{os.path.basename(sid)}.json"
    if os.path.exists(path):
        return json.load(open(path))
    raise HTTPException(404, "unknown session")


@app.websocket("/ws")
async def ws(sock: WebSocket, mode: str = "synthetic", seed: int = 2):  # layout 2: the prior misses first
    await sock.accept()
    if len(sessions) >= MAX_SESSIONS:
        await sock.close(code=1013, reason="server busy")
        return
    sid = f"{mode[:3]}-{uuid.uuid4().hex[:8]}"
    ctrl = sessions[sid] = Controller(CFG, session=sid)
    await sock.send_text(json.dumps({"type": "hello", "session": sid}))
    try:
        if mode == "camera":
            await camera_loop(sock, sid, ctrl)
        else:
            await synthetic_loop(sock, sid, ctrl, seed)
    except WebSocketDisconnect:
        pass
    finally:
        save_trace(sid, sessions.pop(sid))


async def camera_loop(sock, sid, ctrl):
    """Client sends [float64 timestamp][JPEG] per frame and waits for our reply (natural backpressure)."""
    trials = 0
    while True:
        msg = await sock.receive()
        if msg.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect()
        if msg.get("text"):
            if json.loads(msg["text"]).get("cmd") == "restart":
                ctrl = sessions[sid] = Controller(CFG, session=sid)
            continue
        data = msg.get("bytes") or b""
        if not 8 < len(data) < MAX_FRAME:
            continue
        t = struct.unpack("<d", data[:8])[0]
        img = cv.imdecode(np.frombuffer(data[8:], np.uint8), cv.IMREAD_COLOR)
        if img is None:
            continue
        await asyncio.to_thread(ctrl.step, img, t)
        await sock.send_text(packet(ctrl, img))
        if len(ctrl.trials) != trials:
            trials = len(ctrl.trials)
            await asyncio.to_thread(save_trace, sid, ctrl, ctrl.annotate(img))


async def synthetic_loop(sock, sid, ctrl, seed):
    """Server-side tabletop: the browser's buttons are the human's hands (or 'auto' follows instructions)."""
    L = layout(seed)
    world = World(seed=seed, target=L["target"], obstacles=L["obstacles"], guide=L["guide"], truth=L["truth"])
    cmds, auto, trials = asyncio.Queue(), False, 0

    async def reader():
        while True:
            try:
                m = json.loads(await sock.receive_text())
            except ValueError:
                continue
            if isinstance(m, dict) and m.get("cmd") in COMMANDS:
                await cmds.put(m)

    task = asyncio.create_task(reader())
    try:
        while not task.done():
            t0 = time.perf_counter()
            while not cmds.empty():
                m = cmds.get_nowait()
                c, note = m["cmd"], None
                if c == "set":  # the browser dragged a piece: validate before it touches the world or the logs
                    pose, src, obj = nums(m, "x", "y", "a"), nums(m, "fx", "fy"), m.get("obj")
                    if not (pose and src and obj in PIECES):
                        continue
                    pose = on_board(*pose)
                    ctrl.event("human_command", cmd=c, obj=obj, to=[round(v) for v in pose])
                else:
                    ctrl.event("human_command", cmd=c)
                if c == "auto":
                    auto = not auto
                elif c == "occlude":
                    world.occluded = not world.occluded
                elif c in ("new_layout", "restart"):
                    seed = seed + 1 if c == "new_layout" else seed
                    L = layout(seed)
                    world = World(seed=seed, target=L["target"], obstacles=L["obstacles"], guide=L["guide"], truth=L["truth"])
                    if c == "restart":
                        ctrl = sessions[sid] = Controller(CFG, session=sid)
                elif world.traj is not None:
                    note = "Wait for the ball to stop first."
                elif c == "set" and obj == "guide":
                    world.guide = list(pose)
                elif c == "set" and obj == "target":
                    world.target = pose[:2]
                elif c == "set" and world.obstacles:  # the block nearest to where the drag started
                    o = min(world.obstacles, key=lambda o: math.hypot(o.x - src[0], o.y - src[1]))
                    o.x, o.y, o.angle = pose
                elif c == "move_obstacle" and world.obstacles:
                    o = world.obstacles[0]  # bounce around the middle so repeated presses stay on the board
                    world.move_obstacle(0, dx=-70 if o.x > CFG.board_w / 2 else 70, dy=-40 if o.y > CFG.board_h / 2 else 40)
                elif c == "place" and ctrl.plan:
                    world.place_guide(*ctrl.plan["guide"])
                elif c == "release" and ctrl.state != "ready":
                    note = "Place the guide first: the ball is released once the camera has verified the placement."
                elif c == "release":
                    await sock.send_text(json.dumps({"type": "note", "text": "Rolling the ball (Newton rigid-body reality)..."}))
                    await asyncio.to_thread(world.release)
                elif c == "reset_ball":
                    world.reset_ball()
                if note:
                    await sock.send_text(json.dumps({"type": "note", "text": note}))
            for _ in range(2):  # perceive every 30 Hz frame, stream every other one (15 fps is plenty for a browser)
                world.tick()
                img = world.render()
                await asyncio.to_thread(ctrl.step, img, world.t)
            if auto and world.traj is None and not world.occluded and ctrl.state in ("placing", "ready", "reset"):
                if ctrl.state == "ready":
                    await asyncio.to_thread(world.act, ctrl.state, None)
                elif int(world.t * 15) % 6 == 0:
                    world.act(ctrl.state, (ctrl.plan or {}).get("guide"))
            await sock.send_text(packet(ctrl, img, {"auto": auto, "occluded": world.occluded, "seed": seed, "rolling": world.traj is not None}))
            if len(ctrl.trials) != trials:
                trials = len(ctrl.trials)
                await asyncio.to_thread(save_trace, sid, ctrl, ctrl.annotate(img))
            await asyncio.sleep(max(0.0, 2 / 30 - (time.perf_counter() - t0)))
    finally:
        task.cancel()
        sessions[sid] = ctrl


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "web"), html=True), name="web")
