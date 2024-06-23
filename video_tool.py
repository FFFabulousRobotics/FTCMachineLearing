from enum import IntEnum

import aiofiles.os
import aioshutil
from quart import Quart
from common import BackendError, DictProxy, ReturnResult, DATA_FOLDER, ensure_directory
from uuid import uuid4
from pathlib import Path
from typing import Tuple, List, Union
import json
import pandas as pd
import asyncio
import aiofiles
import cv2

class Video:

  class ProcessStatus(IntEnum):
    PREPARING = 0
    PROCESSING = 1
    COMPLETED = 2
    CANCELLED = -1

  def __init__(self, name: str, identifier: Union[str, None] = None):
    self.name = name
    self.identifier = identifier if identifier else uuid4().hex

  async def frame_extract(self, tmp_file_path: str):
    try:
      self.process_status = Video.ProcessStatus.PREPARING
      video = cv2.VideoCapture(tmp_file_path)
      frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
      self.total_frame_count = frame_count
      self.extracted_frame_count = 0
      self.process_status = Video.ProcessStatus.PROCESSING
      ensure_directory(Path(DATA_FOLDER, "videos", self.identifier))
      for i in range(frame_count):
        ret, frame = video.read()
        if not ret:
          break
        cv2.imwrite(
            str(
                Path(DATA_FOLDER, "videos", self.identifier,
                     f"frame_{i+1}.png").absolute()), frame)
        self.extracted_frame_count += 1
      self.resolution = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
      self.labels = pd.DataFrame(columns=[
          "frame_index", "label", "left", "top", "right", "bottom",
          "absolute_left", "absolute_top", "absolute_right", "absolute_bottom"
      ])
      self.excluded_frames = []
      self.process_status = Video.ProcessStatus.COMPLETED
    except asyncio.CancelledError:
      self.process_status = Video.ProcessStatus.CANCELLED
      await aioshutil.rmtree(Path(DATA_FOLDER, "videos", self.identifier))
    except Exception as e:
      self.process_status = Video.ProcessStatus.CANCELLED
      await aioshutil.rmtree(Path(DATA_FOLDER, "videos", self.identifier))
    finally:
      print("Video frame extract finished: " + self.identifier)
      video.release()
      await aiofiles.os.remove(tmp_file_path)

  def start_frame_extract(self, tmp_file_path: str):
    coro = self.frame_extract(tmp_file_path)
    self.frame_extract_task = asyncio.ensure_future(coro)
    asyncio.run_coroutine_threadsafe(coro=coro, loop=asyncio.get_event_loop())

  def exclude_frame(self, frame_index: int):
    self.excluded_frames.append(frame_index)

  def frame_extract_finished(self):
    if not hasattr(self, "process_status"):
      return False
    return self.process_status == Video.ProcessStatus.COMPLETED \
      or self.process_status == Video.ProcessStatus.CANCELLED

  def label_frame(self, frame_index: int, label: str, box: Tuple[int, int, int,
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
                            trackers: List[cv2.Tracker], labels: List[str]):
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
      frame_path = Path(DATA_FOLDER, "videos", self.identifier,
                        f"frame_{frame_index}.png")
      frame = cv2.imread(str(frame_path.absolute()))
      raw_bboxes = self.labels["frame_index", "label",
                            "absolute_left", "absolute_top", "absolute_right", "absolute_bottom"] \
        .loc[self.labels["frame_index"] == frame_index]
      bboxes = raw_bboxes["absolute_left", "absolute_top",
                          "absolute_right", "absolute_bottom"].to_dict(
                              orient="records")  # type: ignore
      labels = raw_bboxes["label"].tolist()
      trackers = []
      for bbox in bboxes:
        if algorithm == "KCF":
          tracker = cv2.TrackerKCF()
        elif algorithm == "MedianFlow":
          tracker = cv2.legacy.TrackerMedianFlow()
        elif algorithm == "MOSSE":
          tracker = cv2.legacy.TrackerMOSSE()
        elif algorithm == "CSRT":
          tracker = cv2.TrackerCSRT()
        elif algorithm == "MIL":
          tracker = cv2.TrackerMIL()
        elif algorithm == "TLD":
          tracker = cv2.legacy.TrackerTLD()
        elif algorithm == "Boosting":
          tracker = cv2.legacy.TrackerBoosting()
        else:
          raise ValueError(f"Unknown tracker algorithm: {algorithm}")

        trackers.append(tracker)
        tracker.init(frame, (bbox["absolute_left"], bbox["absolute_top"],
                             bbox["absolute_right"] - bbox["absolute_left"],
                             bbox["absolute_bottom"] - bbox["absolute_top"]))
      frame_index += 1
      while frame_index <= self.total_frame_count:
        frame_path = Path(DATA_FOLDER, "videos", self.identifier,
                          f"frame_{frame_index}.png")
        frame = cv2.imread(str(frame_path.absolute()))
        result = await self.track_one_frame(frame_index, frame, trackers,
                                            labels)
        if not result.is_success:
          return result
        self.labels = pd.concat(
            [self.labels, pd.DataFrame(result.data)], ignore_index=True)
        continue_event.clear()
        await continue_event.wait()
        frame_index += 1
      return ReturnResult.success()
    except asyncio.CancelledError:
      return ReturnResult(BackendError.TASK_WAS_CANCELLED)

  def start_object_tracking(self, starting_frame_index: int, algorithm: str,
                            continue_event: asyncio.Event):
    coro = self.track_from_frame(starting_frame_index, algorithm,
                                 continue_event)
    self.track_task = asyncio.ensure_future(coro)
    asyncio.run_coroutine_threadsafe(coro=coro, loop=asyncio.get_event_loop())
    return self.track_task

  @property
  def info(self):
    if self.frame_extract_finished():
      labeled_frame_count = len(
          self.labels.drop_duplicates(subset=["frame_index"]))
      excluded_frame_count = len(self.excluded_frames)
      return self.name, self.resolution, self.total_frame_count, labeled_frame_count, excluded_frame_count
    else:
      return self.name, self.process_status, self.total_frame_count, self.extracted_frame_count

  def to_dict(self):
    if self.frame_extract_finished():
      self.save_labels()
      return ReturnResult.success({
          "name": self.name,
          "identifier": self.identifier,
          "resolution": self.resolution,
          "total_frame_count": self.total_frame_count,
          "excluded_frames": self.excluded_frames
      })
    else:
      return ReturnResult(BackendError.VIDEO_PROCESSING)

  def save_labels(self):
    self.labels.to_csv(Path(DATA_FOLDER, "videos", self.identifier,
                            "labels.csv"),
                       index=False)

  def load_labels(self):
    if Path(DATA_FOLDER, "videos", self.identifier, "labels.csv").exists():
      self.labels = pd.read_csv(
          Path(DATA_FOLDER, "videos", self.identifier, "labels.csv"))
    else:
      self.labels = pd.DataFrame(columns=[
          "frame_index", "label", "left", "top", "right", "bottom",
          "absolute_left", "absolute_top", "absolute_right", "absolute_bottom"
      ])

  @classmethod
  def from_dict(cls, d: dict):
    if "name" not in d or "identifier" not in d or "resolution" not in d or "total_frame_count" not in d or "excluded_frames" not in d:
      return ReturnResult(BackendError.INVALID_ARGUMENT)
    video = cls(d["name"], d["identifier"])
    video.resolution = d["resolution"]
    video.total_frame_count = d["total_frame_count"]
    video.excluded_frames = d["excluded_frames"]
    video.process_status = Video.ProcessStatus.COMPLETED
    video.frame_extract_task = None
    video.track_task = None
    video.load_labels()
    return ReturnResult.success(video)

videos: DictProxy[str, Video] = DictProxy()
frame_ids: DictProxy[str, str] = DictProxy()

async def upload_video(name: str, tmp_file_path: str):
  video = Video(name)
  await videos.put(video.identifier, video)
  video.start_frame_extract(tmp_file_path)
  return ReturnResult.success(name, video.identifier)

async def get_video_info(video_identifier: str):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = await videos[video_identifier]
  return ReturnResult.success(video.frame_extract_finished(), *video.info)

async def get_all_videos():
  video_infos = {}
  for video in await videos.values():
    video_infos[video.identifier] = (video.frame_extract_finished(),
                                     *video.info)
  return ReturnResult.success(video_infos)

async def cancel_process(video_identifier: str):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = await videos[video_identifier]
  video.frame_extract_task.cancel()
  return ReturnResult.success()

async def cleanup_frame_cache(frame_id):
  await asyncio.sleep(10)
  await aiofiles.os.remove(Path("static", "img", "tmp", f"{frame_id}.png"))

async def read_frame(video_identifier: str, frame_index: int):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = await videos[video_identifier]
  if video.total_frame_count < frame_index:
    return ReturnResult(BackendError.FRAME_NOT_FOUND)
  if not video.frame_extract_finished():
    return ReturnResult(BackendError.VIDEO_PROCESSING)
  frame_path = Path(DATA_FOLDER, "videos", video_identifier,
                    f"frame_{frame_index}.png")
  frame_id = uuid4().hex
  await aioshutil.copyfile(frame_path,
                           Path("static", "img", "tmp", f"{frame_id}.png"))
  frame_ids[frame_id] = f"img/tmp/{frame_id}.png"
  asyncio.create_task(cleanup_frame_cache(frame_id))
  frame_labels = video.labels \
    ["label_id", "frame_index", "label", \
    "absolute_left", "absolute_top", "absolute_right", "absolute_bottom"] \
    .loc[video.labels["frame_index"] == frame_index].to_dict(orient="records") # type: ignore
  return ReturnResult.success(frame_id, frame_labels)

def get_frame_png(frame_id: str):
  if frame_id not in frame_ids:
    return ReturnResult(BackendError.FRAME_NOT_FOUND)
  frame_path = frame_ids[frame_id]
  del frame_ids[frame_id]
  return ReturnResult.success(frame_path)

async def label_frame(video_identifier: str, frame_index: int, label: str,
                      box: Tuple[int, int, int, int]):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = await videos[video_identifier]
  return video.label_frame(frame_index, label, box)

async def unlabel_frame(video_identifier: str, label_id: str):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = await videos[video_identifier]
  return video.unlabel_frame(label_id)

async def exclude_frame(video_identifier: str, frame_index: int):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = await videos[video_identifier]
  video.exclude_frame(frame_index)
  return ReturnResult.success()

async def start_object_tracking(video_identifier: str, start_frame_index: int,
                                algorithm: str):
  if video_identifier not in videos:
    return ReturnResult(BackendError.VIDEO_NOT_FOUND)
  video = await videos[video_identifier]
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
  task = video.start_object_tracking(start_frame_index, algorithm,
                                     continue_event)
  return ReturnResult.success(task, continue_event)

async def save_videos():
  json_dict = []
  for video in await videos.values():
    if (vresult := video.to_dict()).is_success:
      json_dict.append(vresult.data)
    else:
      print(f"Failed to save video {video.identifier}")
  with open(Path(DATA_FOLDER, "videos.json"), "w") as f:
    json.dump(json_dict, f)
  print(f"Saved {len(json_dict)} videos")

async def load_videos():
  if not Path(DATA_FOLDER, "videos.json").exists():
    return
  with open(Path(DATA_FOLDER, "videos.json"), "r") as f:
    videos_json = json.load(f)
  for video_json in videos_json:
    vresult = Video.from_dict(video_json)
    if vresult.is_success:
      await videos.put(vresult.data.identifier, vresult.data)
    else:
      print(f"Failed to load video {video_json['identifier']}")
  print(f"Loaded {len(videos)} videos")
