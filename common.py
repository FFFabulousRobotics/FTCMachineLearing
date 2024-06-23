from enum import IntEnum
from pathlib import Path
from asyncio import Lock
import tempfile
from typing import Generic, TypeVar, Dict
from typing_extensions import override

DATA_FOLDER = "data"
TEMP_FOLDER = Path(tempfile.gettempdir(), "FTCMachineLearning").absolute()

__necessary_directories = (Path(DATA_FOLDER), Path(DATA_FOLDER,
                                                   "videos"), TEMP_FOLDER)

def ensure_directory(path: Path):
  path.mkdir(parents=True, exist_ok=True)

def ensure_directories():
  for directory in __necessary_directories:
    ensure_directory(directory)

class BackendError(IntEnum):
  SUCCESS = 0
  VIDEO_NOT_FOUND = 1
  VIDEO_PROCESSING = 2
  FRAME_NOT_FOUND = 3
  NO_MORE_FRAMES = 4
  UNACCEPTABLE_ALGORITHM = 5
  NO_FILE_UPLOADED = 6
  UNSUPPORTED_FILE_TYPE = 7
  INVALID_REQUEST = 8
  TRACKING_FAILED = 9
  FRAME_NOT_LABELED = 10
  TASK_WAS_CANCELLED = 11
  INVALID_ARGUMENT = 12

  @property
  def error_message(self):
    if self == BackendError.SUCCESS:
      return "Success"
    elif self == BackendError.VIDEO_NOT_FOUND:
      return "Video not found"
    elif self == BackendError.VIDEO_PROCESSING:
      return "Frame extracting for this video is still ongoing"
    elif self == BackendError.FRAME_NOT_FOUND:
      return "Frame not found"
    elif self == BackendError.NO_MORE_FRAMES:
      return "No more frames for object tracking"
    elif self == BackendError.UNACCEPTABLE_ALGORITHM:
      return "Unacceptable algorithm for object tracking"
    elif self == BackendError.NO_FILE_UPLOADED:
      return "No file uploaded"
    elif self == BackendError.UNSUPPORTED_FILE_TYPE:
      return "Unsupported file type"
    elif self == BackendError.INVALID_REQUEST:
      return "Invalid request, check your request body"
    elif self == BackendError.TRACKING_FAILED:
      return "Object tracking failed"
    elif self == BackendError.FRAME_NOT_LABELED:
      return "Frame hasn't labeled, can't start object tracking"
    elif self == BackendError.TASK_WAS_CANCELLED:
      return "Task was cancelled"
    elif self == BackendError.INVALID_ARGUMENT:
      return "Invalid argument"
    else:
      return "Unknown error"

class ReturnResult:

  def __init__(self, status: BackendError, *data):
    self.status = status
    self.message = status.error_message
    self._data = data

  @property
  def is_success(self):
    return self.status == BackendError.SUCCESS
  
  @property
  def data(self):
    if len(self._data) == 1:
      return self._data[0]
    else:
      return self._data

  @classmethod
  def success(cls, *data):
    return cls(BackendError.SUCCESS, *data)

K = TypeVar('K')
V = TypeVar('V')

class DictProxy(Generic[K, V], Dict[K, V]):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.lock = Lock()

  async def __getitem__(self, key: K):
    async with self.lock:
      return super().__getitem__(key)

  async def __setitem__(self, key: K, value: V):
    async with self.lock:
      return super().__setitem__(key, value)

  async def put(self, key: K, value: V):
    return await self.__setitem__(key, value)
  
  async def values(self):
    async with self.lock:
      return super().values()
