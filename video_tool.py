from multiprocessing.managers import DictProxy
from multiprocessing import Manager
from enum import IntEnum
from common import BackendError, ReturnResult, DATA_FOLDER
from uuid import uuid4
import pandas as pd
import shutil
import asyncio
import cv2
import os

class Video:

  class ProcessStatus(IntEnum):
    PREPARING = 0
    PROCESSING = 1
    COMPLETED = 2
    CANCELLED = -1

  def __init__(self, tmp_file_path: str):
    self.tmp_file_path = tmp_file_path
    self.identifier = uuid4().hex

  async def frame_extract(self):
    try:
      self.process_status = Video.ProcessStatus.PREPARING
      video = cv2.VideoCapture(self.tmp_file_path)
      frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
      self.total_frame_count = frame_count
      self.extracted_frame_count = 0
      self.process_status = Video.ProcessStatus.PROCESSING
      for i in range(frame_count):
        ret, frame = video.read()
        if not ret:
          break
        cv2.imwrite(
            os.path.join(DATA_FOLDER, "video", self.identifier,
                         f"frame_{i+1}.png"), frame)
        self.extracted_frame_count += 1
      self.resolution = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
      self.labels = pd.DataFrame(columns=[
          "frame_index", "label", "left", "top", "right", "bottom",
          "absolute_left", "absolute_top", "absolute_right", "absolute_bottom"
      ])
      self.process_status = Video.ProcessStatus.COMPLETED
    except asyncio.CancelledError:
      self.process_status = Video.ProcessStatus.CANCELLED
      shutil.rmtree(os.path.join(DATA_FOLDER, "video", self.identifier))
    finally:
      video.release()
      os.remove(self.tmp_file_path)

  def start_frame_extract(self):
    self.coro_task = asyncio.create_task(self.frame_extract())

  def frame_extract_finished(self):
    return self.process_status == Video.ProcessStatus.COMPLETED \
      or self.process_status == Video.ProcessStatus.CANCELLED

  def label_frame(self, frame_index: int, label: str, box: tuple[int, int, int,
                                                                 int]):
    if self.total_frame_count < frame_index:
      return ReturnResult(BackendError.FRAME_NOT_FOUND)
    if not self.frame_extract_finished():
      return ReturnResult(BackendError.VIDEO_PROCESSING)
    label_id = uuid4().hex
    record = {
        "label_id": label_id,
        "frame_index": frame_index,
        "label": label,
        "left": box[0] / self.resolution[0],
        "top": box[1] / self.resolution[1],
        "right": box[2] / self.resolution[0],
        "bottom": box[3] / self.resolution[1],
        "absolute_left": box[0],
        "absolute_top": box[1],
        "absolute_right": box[2],
        "absolute_bottom": box[3]
    }
    new_df = pd.DataFrame([record])
    self.labels = pd.concat([self.labels, new_df], ignore_index=True)
    return ReturnResult.success(label_id)
  
  def unlabel_frame(self, label_id: str):
    self.labels = self.labels.loc[self.labels["label_id"] != label_id]
    return ReturnResult.success()

  async def track_one_frame(self, new_frame_index: int,
                            new_frame: cv2.typing.MatLike,
                            trackers: list[cv2.Tracker],
                            labels: list[str]):
    result = []
    for i, tracker in enumerate(trackers):
      success, bbox = tracker.update(new_frame)
      if success:
        al, at, w, h = bbox
        ar = al + w
        ab = at + h
        label_id = uuid4().hex
        result.append({
            "label_id": label_id,
            "frame_index": new_frame_index,
            "label": labels[i],
            "left": al / self.resolution[0],
            "top": at / self.resolution[1],
            "right": ar / self.resolution[0],
            "bottom": ab / self.resolution[1],
            "absolute_left": al,
            "absolute_top": at,
            "absolute_right": ar,
            "absolute_bottom": ab
        })
      else:
        return ReturnResult(BackendError.TRACKING_FAILED)
    return ReturnResult.success(result)

  async def track_from_frame(self, starting_frame_index: int, algorithm: str,
                             continue_event: asyncio.Event):
    try:
      frame_index = starting_frame_index
      frame_path = os.path.join(DATA_FOLDER, "video", self.identifier,
                                f"frame_{frame_index}.png")
      frame = cv2.imread(frame_path)
      raw_bboxes = self.labels["frame_index", "label",
                            "absolute_left", "absolute_top", "absolute_right", "absolute_bottom"] \
        .loc[self.labels["frame_index"] == frame_index]
      bboxes = raw_bboxes["absolute_left", "absolute_top", "absolute_right", "absolute_bottom"].to_dict(orient="records")
      labels = raw_bboxes["label"].array
      trackers = []
      for bbox in bboxes:
        match algorithm:
          case "KCF":
            tracker = cv2.TrackerKCF()
          case "MedianFlow":
            tracker = cv2.legacy.TrackerMedianFlow()
          case "MOSSE":
            tracker = cv2.legacy.TrackerMOSSE()
          case "CSRT":
            tracker = cv2.TrackerCSRT()
          case "MIL":
            tracker = cv2.TrackerMIL()
          case "TLD":
            tracker = cv2.legacy.TrackerTLD()
          case "Boosting":
            tracker = cv2.legacy.TrackerBoosting()
        trackers.append(tracker)
        tracker.init(frame, (bbox["absolute_left"], bbox["absolute_top"],
                             bbox["absolute_right"] - bbox["absolute_left"],
                             bbox["absolute_bottom"] - bbox["absolute_top"]))
      frame_index += 1
      while frame_index <= self.total_frame_count:
        frame_path = os.path.join(DATA_FOLDER, "video", self.identifier,
                                  f"frame_{frame_index}.png")
        frame = cv2.imread(frame_path)
        result = await self.track_one_frame(frame_index, frame, trackers, labels)
        if not result.is_success():
          return result
        self.labels = pd.concat(
            [self.labels, pd.DataFrame(result.data)], ignore_index=True)
        continue_event.clear()
        continue_event.wait()
        frame_index += 1
      return ReturnResult.success()
    except asyncio.CancelledError:
      return ReturnResult(BackendError.TASK_WAS_CANCELLED)

  @property
  def info(self):
    if self.frame_extract_finished():
      labeled_frame_count = len(
          self.labels.drop_duplicates(subset=["frame_index"]))
      return self.resolution, self.total_frame_count, labeled_frame_count
    else:
      return self.process_status, self.total_frame_count, self.extracted_frame_count

video_tool_manager = Manager()
videos: DictProxy[str, Video] = video_tool_manager.dict()
frame_ids: DictProxy[str, str] = video_tool_manager.dict()

def upload_video(tmp_file_path: str):
  video = Video(tmp_file_path)
  videos[video.identifier] = video
  video.start_frame_extract()
  return ReturnResult.success(video.identifier)

def get_video_info(video_identifier: str):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = videos[video_identifier]
  return ReturnResult.success(video.frame_extract_finished(), *video.info)

def cancel_process(video_identifier: str):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = videos[video_identifier]
  video.coro_task.cancel()
  return ReturnResult.success()

def read_frame(video_identifier: str, frame_index: int):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = videos[video_identifier]
  if video.total_frame_count < frame_index:
    return ReturnResult(BackendError.FRAME_NOT_FOUND)
  if not video.frame_extract_finished():
    return ReturnResult(BackendError.VIDEO_PROCESSING)
  frame_path = os.path.join(DATA_FOLDER, "video", video_identifier,
                            f"frame_{frame_index}.png")
  frame_id = uuid4().hex
  frame_ids[frame_id] = frame_path
  frame_labels = video.labels \
    ["label_id", "frame_index", "label", \
    "absolute_left", "absolute_top", "absolute_right", "absolute_bottom"] \
    .loc[video.labels["frame_index"] == frame_index].to_dict(orient="records")
  return ReturnResult.success(frame_id, frame_labels)

def get_frame_png(frame_id: str):
  if frame_id not in frame_ids:
    return ReturnResult(BackendError.FRAME_NOT_FOUND)
  frame_path = frame_ids[frame_id]
  del frame_ids[frame_id]
  return ReturnResult.success(frame_path)

def label_frame(video_identifier: str, frame_index: int, label: str,
                box: tuple[int, int, int, int]):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = videos[video_identifier]
  return video.label_frame(frame_index, label, box)

def unlabel_frame(video_identifier: str, label_id: str):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = videos[video_identifier]
  return video.unlabel_frame(label_id)

def start_object_tracking(video_identifier: str, start_frame_index: int,
                          algorithm: str):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = videos[video_identifier]
  if video.total_frame_count < start_frame_index:
    return ReturnResult(BackendError.FRAME_NOT_FOUND)
  if video.total_frame_count == start_frame_index:
    return ReturnResult(BackendError.NO_MORE_FRAMES)
  if not video.frame_extract_finished():
    return ReturnResult(BackendError.VIDEO_PROCESSING)
  acceptable_algorithms = [
      "KCF", "MedianFlow", "MOSSE", "CSRT", "MIL", "TLD", "Boosting"
  ]
  if algorithm not in acceptable_algorithms:
    return ReturnResult(BackendError.UNACCEPTABLE_ALGORITHM)
  if start_frame_index not in video.labels["frame_index"].array:
    return ReturnResult(BackendError.FRAME_NOT_LABELED)
  continue_event = asyncio.Event()
  task = asyncio.create_task(
      video.track_from_frame(start_frame_index, algorithm, continue_event))
  return ReturnResult.success(task, continue_event)
