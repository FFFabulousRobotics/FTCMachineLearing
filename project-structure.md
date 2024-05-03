# Project Structure

> Special notations used in this document:  
> **\[async\]**: The method is an async method.  
> **A-desc**, **B-desc**: Optional descriptions. A-descs often give a general idea of what this class/method is/does, while B-descs give detailed explanations of the class/method.

## Video Tool

### Video class

#### Video.ProcessStatus enum

|Name|Real Value|
|-|-|
|PREPARING|0|
|PROCESSING|1|
|COMPLETED|2|
|CANCELLED|-1|

#### methods

- \_\_init\_\_
  - **params**:
    - self
    - tmp_file_path: str
      - The file path for the actual video file. The file will be deleted so it's temporary.
  - **B-desc**: The constructor method uses the video file's MD5 value to assign the video a unique identifier.
- frame_extract \[async\]
  - **A-desc**: Extract frames frooom the video file.
  - **B-desc**: This method uses opencv to read the total frame count, frame images and video resolution. The total frame count and the resolution are stored in self.total_frame_count and self.resolution respectively. self.status can be checked to monitor the process.
- 