from asyncio import Task
from threading import Event
from flask import Flask, request, render_template
from flask_sock import Server, Sock
from json import dumps, loads
from pathlib import Path
from common import BackendError
from time import sleep
import tempfile
import video_tool

app = Flask(__name__)
sock = Sock(app)
ALLOWED_EXTENSIONS = {'.mp4'}

# web page routes

@app.route('/')
@app.route('/index')
@app.route('/index.html')
def index_page():
  return render_template('index.html')

@app.route('/js/<path:path>')
def send_js(path):
  return app.send_static_file('static/js/' + path)

@app.route('/css/<path:path>')
def send_css(path):
  return app.send_static_file('static/css/' + path)

# api routes

@app.route('/api')
def api_root():
  welcome = {"welcome": "Welcome to the FTC Machine Learning API", "status": 0}
  return dumps(welcome)

@app.route("/api/video")
def api_video_root():
  response = {"status": 0}
  return dumps(response)

@app.route("/api/video/<string:video_id>", methods=['GET'])
def api_video_info(video_id):
  result = video_tool.get_video_info(video_id)
  if not result.is_success():
    return dumps({"status": result.status, "error": result.message})
  processed, data = result.data[0], result.data[1:]
  if processed:
    return dumps({
        "status": 0,
        "video_type": "processed",
        "video_resolution": data[0],
        "total_frame_count": data[1],
        "labeled_frame_count": data[2]
    })
  return dumps({
      "status": 0,
      "video_type": "processing",
      "video_status": data[0],
      "total_frame_count": data[1],
      "extracted_frame_count": data[2]
  })

@app.route('/api/video/upload', methods=['POST'])
def api_video_upload():
  nfu = BackendError.NO_FILE_UPLOADED
  uft = BackendError.UNSUPPORTED_FILE_TYPE
  if 'file' not in request.files:
    return dumps({"status": nfu, "error": nfu.error_message})
  file = request.files['file']
  if not file:
    return dumps({"status": nfu, "error": nfu.error_message})
  if file.filename == '':
    return dumps({"status": nfu, "error": nfu.error_message})
  if not '.' in file.filename and Path(
      file.filename).suffix in ALLOWED_EXTENSIONS:
    return dumps({"status": uft, "error": uft.error_message})
  tmp_file_path = tempfile.mkstemp(suffix='.mp4')[1]
  with open(tmp_file_path, 'wb') as tmp_file:
    file.save(tmp_file)
  video_id = video_tool.upload_video(tmp_file_path).data[0]
  return dumps({"status": 0, "video_id": video_id})

@app.route('/api/video/<string:video_id>/cancel', methods=['GET'])
def api_frame_extract_cancel(video_id):
  result = video_tool.cancel_process(video_id)
  if not result.is_success():
    return dumps({"status": result.status, "error": result.message})
  return dumps({"status": 0})

@app.route('/api/video/<string:video_id>/frames/<int:index>', methods=['GET'])
def api_read_frame(video_id, index):
  result = video_tool.read_frame(video_id, index)
  if not result.is_success():
    return dumps({"status": result.status, "error": result.message})
  return dumps({
      "status": 0,
      "frame_id": result.data[0],
      "frame_labels": result.data[1]
  })

@app.route('/api/frame/<string:frame_id>', methods=['GET'])
def api_get_frame_png(frame_id):
  result = video_tool.get_frame_png(frame_id)
  if not result.is_success():
    return app.send_static_file('static/img/placeholder.png')
  return app.send_static_file(result.data[0])

@app.route('/api/video/<string:video_id>/frames/<int:index>/label',
           methods=['POST'])
def api_label_frame(video_id, index):
  if 'label' not in request.json or 'box' not in request.json:
    ir = BackendError.INVALID_REQUEST
    return dumps({"status": ir, "error": ir.error_message})
  label = request.json['label']
  box = request.json['box']
  result = video_tool.label_frame(video_id, index, label, box)
  if not result.is_success():
    return dumps({"status": result.status, "error": result.message})
  return dumps({"status": 0, "label_id": result.data[0]})

@app.route('/api/video/<string:video_id>/frames/<int:index>/unlabel',
           methods=['POST'])
def api_unlabel_frame(video_id, label_id):
  result = video_tool.unlabel_frame(video_id, label_id)
  if not result.is_success():
    return dumps({"status": result.status, "error": result.message})
  return dumps({"status": 0})

@sock.route('/api/video/<string:video_id>/object_tracking')
def api_object_tracking(ws: Server, video_id):
  params = ws.receive()
  if not params or (type(params) != str and type(params) != bytes):
    ws.close(1007)
    return
  params = loads(params)
  if 'start' not in params or 'algorithm' not in params:
    ws.close(1007)
    return
  frame_index = params['start']
  algorithm = params['algorithm']
  result = video_tool.start_object_tracking(video_id, frame_index, algorithm)
  if not result.is_success():
    match result.status:
      case BackendError.VIDEO_NOT_FOUND | \
            BackendError.VIDEO_PROCESSING | \
            BackendError.FRAME_NOT_FOUND | \
            BackendError.NO_MORE_FRAMES | \
            BackendError.FRAME_NOT_LABELED:
        code = 1011
      case BackendError.UNACCEPTABLE_ALGORITHM:
        code = 1008
      case _:
        code = 1006
    ws.close(code, result.message)
  task: Task = result.data[0]
  continue_event: Event = result.data[1]
  while not task.done():
    while continue_event.is_set():
      sleep(0.1)
    frame_index += 1
    frame_result = video_tool.read_frame(video_id, frame_index)
    if not frame_result.is_success():
      ws.close(1011, frame_result.message)
      task.cancel()
      return
    frame_id, frame_labels = frame_result.data
    ws.send(dumps({"frame_id": frame_id, "frame_labels": frame_labels}))
    continue_event.set()
  ws.close()
