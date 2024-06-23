from asyncio import Task
from threading import Event
from quart import Quart, request, render_template, websocket as ws
from quart.datastructures import FileStorage
from json import dumps, loads
from pathlib import Path
from common import TEMP_FOLDER, BackendError, ensure_directories
from collections import namedtuple
from time import sleep
import os.path
import tempfile
import video_tool

app = Quart(__name__)
ALLOWED_EXTENSIONS = {'.mp4'}

ensure_directories()

VideoInfo = namedtuple('VideoInfo', [
    'id', 'name', 'resolution', 'status', 'total_frames', 'extracted_frames',
    'labeled_frames', 'excluded_frames'
])

@app.before_serving
async def before_serving():
  await video_tool.load_videos()

@app.after_serving
async def after_serving():
  await video_tool.save_videos()

# web page routes

@app.route('/')
@app.route('/index')
@app.route('/index.html')
async def index_page():
  videos_result = await video_tool.get_all_videos()
  if not videos_result.is_success:
    return await render_template('index.html.j2', videos=[])
  video_infos: dict = videos_result.data
  videos = []
  for video_id, video_info in video_infos.items():
    if video_info[0]:
      videos.append(
          VideoInfo(video_id, video_info[1], video_info[2], "PROCESSED",
                    video_info[3], video_info[3], video_info[4],
                    video_info[5]))
    else:
      videos.append(
          VideoInfo(video_id, video_info[1], "-",
                    video_info[2].name.split(".")[1], video_info[3],
                    video_info[4], "-", "-"))
  return await render_template('index.html.j2', videos=videos)

@app.route('/js/<path:path>')
async def send_js(path):
  if os.path.exists('static/js/' + path):
    return await app.send_static_file('js/' + path)
  else:
    return await render_template('error.html.j2', error_code=404)

@app.route('/css/<path:path>')
async def send_css(path):
  if os.path.exists('static/css/' + path):
    return await app.send_static_file('css/' + path)
  else:
    return await render_template('error.html.j2', error_code=404)

@app.route('/favicon.ico')
async def send_favicon():
  return await app.send_static_file('favicon.ico')

# api routes

@app.route('/api')
async def api_root():
  welcome = {"welcome": "Welcome to the FTC Machine Learning API", "status": 0}
  return dumps(welcome)

@app.route("/api/videos")
async def api_all_videos():
  videos = video_tool.get_all_videos()
  if not videos.is_success:
    return dumps({"status": videos.status, "error": videos.message})
  return dumps({"status": 0, "videos": videos.data})

@app.route("/api/video/<string:video_id>", methods=['GET'])
async def api_video_info(video_id):
  result = await video_tool.get_video_info(video_id)
  if not result.is_success:
    return dumps({"status": result.status, "error": result.message})
  processed, data = result.data[0], result.data[1:]
  if processed:
    return dumps({
        "status": 0,
        "video_type": "processed",
        "video_resolution": data[0],
        "total_frame_count": data[1],
        "labeled_frame_count": data[2],
        "excluded_frame_count": data[3]
    })
  return dumps({
      "status": 0,
      "video_type": "processing",
      "video_status": data[0],
      "total_frame_count": data[1],
      "extracted_frame_count": data[2]
  })

@app.route('/api/video/upload', methods=['POST'])
async def api_video_upload():
  nfu = BackendError.NO_FILE_UPLOADED
  uft = BackendError.UNSUPPORTED_FILE_TYPE
  if 'file' not in await request.files:
    return dumps({"status": nfu, "error": nfu.error_message})
  file: FileStorage = (await request.files)['file']
  if not file:
    return dumps({"status": nfu, "error": nfu.error_message})
  assert isinstance(file, FileStorage)
  if not file.filename:
    return dumps({"status": nfu, "error": nfu.error_message})
  if not '.' in file.filename and Path(
      file.filename).suffix in ALLOWED_EXTENSIONS:
    return dumps({"status": uft, "error": uft.error_message})
  tmp_file_path = tempfile.mkstemp(dir=str(TEMP_FOLDER),
                                   prefix="vid",
                                   suffix='.mp4')[1]
  await file.save(tmp_file_path)  # type: ignore
  file.close()
  if 'name' in await request.form:
    name = (await request.form)['name']
  else:
    name = file.filename.rsplit('.')[0]
  video_name, video_id = (await video_tool.upload_video(name,
                                                        tmp_file_path)).data
  return dumps({"status": 0, "video_id": video_id, "video_name": video_name})

@app.route('/api/video/<string:video_id>/cancel', methods=['GET'])
async def api_frame_extract_cancel(video_id):
  result = await video_tool.cancel_process(video_id)
  if not result.is_success:
    return dumps({"status": result.status, "error": result.message})
  return dumps({"status": 0})

@app.route('/api/video/<string:video_id>/frames/<int:index>', methods=['GET'])
async def api_read_frame(video_id, index):
  result = await video_tool.read_frame(video_id, index)
  if not result.is_success:
    return dumps({"status": result.status, "error": result.message})
  return dumps({
      "status": 0,
      "frame_id": result.data[0],
      "frame_labels": result.data[1]
  })

@app.route('/api/frame/<string:frame_id>', methods=['GET'])
async def api_get_frame_png(frame_id):
  result = video_tool.get_frame_png(frame_id)
  if not result.is_success:
    return await app.send_static_file('img/placeholder.png')
  return await app.send_static_file(result.data)

@app.route('/api/video/<string:video_id>/frames/<int:index>/label',
           methods=['POST'])
async def api_label_frame(video_id, index):
  if 'label' not in await request.json or 'box' not in await request.json:
    ir = BackendError.INVALID_REQUEST
    return dumps({"status": ir, "error": ir.error_message})
  label = (await request.json)['label']
  box = (await request.json)['box']
  result = await video_tool.label_frame(video_id, index, label, box)
  if not result.is_success:
    return dumps({"status": result.status, "error": result.message})
  return dumps({"status": 0, "label_id": result.data})

@app.route('/api/video/<string:video_id>/frames/<int:index>/unlabel',
           methods=['GET'])
async def api_unlabel_frame(video_id, label_id):
  result = await video_tool.unlabel_frame(video_id, label_id)
  if not result.is_success:
    return dumps({"status": result.status, "error": result.message})
  return dumps({"status": 0})

@app.route('/api/video/<string:video_id>/frames/<int:index>/exclude',
           methods=['GET'])
async def api_exclude_frame(video_id, index):
  result = await video_tool.exclude_frame(video_id, index)
  if not result.is_success:
    return dumps({"status": result.status, "error": result.message})
  return dumps({"status": 0})

@app.websocket('/api/video/<string:video_id>/object_tracking')
async def api_object_tracking(video_id: str):
  params = await ws.receive()
  if not params or (type(params) != str and type(params) != bytes):
    await ws.close(1007)
    return
  params = loads(params)
  if 'start' not in params or 'algorithm' not in params:
    await ws.close(1007)
    return
  frame_index = params['start']
  algorithm = params['algorithm']
  result = await video_tool.start_object_tracking(video_id, frame_index,
                                                  algorithm)
  if not result.is_success:
    if result.status in [
        BackendError.VIDEO_NOT_FOUND, BackendError.VIDEO_PROCESSING,
        BackendError.FRAME_NOT_FOUND, BackendError.NO_MORE_FRAMES,
        BackendError.FRAME_NOT_LABELED
    ]:
      code = 1011
    elif result.status == BackendError.UNACCEPTABLE_ALGORITHM:
      code = 1008
    else:
      code = 1006

    await ws.close(code, result.message)
  task: Task = result.data[0]
  continue_event: Event = result.data[1]
  while not task.done():
    while continue_event.is_set():
      sleep(0.1)
    frame_index += 1
    frame_result = await video_tool.read_frame(video_id, frame_index)
    if not frame_result.is_success:
      await ws.close(1011, frame_result.message)
      task.cancel()
      return
    frame_id, frame_labels = frame_result.data
    await ws.send(dumps({"frame_id": frame_id, "frame_labels": frame_labels}))
    continue_event.set()
  await ws.close(1000)
