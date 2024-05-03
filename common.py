from enum import IntEnum

DATA_FOLDER = "data"

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

  @property
  def error_message(self):
    match self:
      case BackendError.SUCCESS:
        return "Success"

      case BackendError.VIDEO_NOT_FOUND:
        return "Video not found"

      case BackendError.VIDEO_PROCESSING:
        return "Frame extracting for this video is still ongoing"

      case BackendError.FRAME_NOT_FOUND:
        return "Frame not found"

      case BackendError.NO_MORE_FRAMES:
        return "No more frames for object tracking"

      case BackendError.UNACCEPTABLE_ALGORITHM:
        return "Unacceptable algorithm for object tracking"

      case BackendError.NO_FILE_UPLOADED:
        return "No file uploaded"
        
      case BackendError.UNSUPPORTED_FILE_TYPE:
        return "Unsupported file type"
      
      case BackendError.INVALID_REQUEST: 
        return "Invalid request, check your request body"\

      case BackendError.TRACKING_FAILED:
        return "Object tracking failed"
      
      case BackendError.FRAME_NOT_LABELED:
        return "Frame hasn't labeled, can't start object tracking"
      
      case BackendError.TASK_WAS_CANCELLED:
        return "Task was cancelled"

class ReturnResult:
  def __init__(self, status: BackendError, *data):
    self.status = status
    self.message = status.error_message
    self.data = data
  
  @property
  def is_success(self):
    return self.status == BackendError.SUCCESS
  
  @classmethod
  def success(cls, *data):
    return cls(BackendError.SUCCESS, *data)