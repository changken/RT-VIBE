# Overview
This is a wrapper of VIBE at [https://github.com/mkocabas/VIBE]

Under development.

# Task list
- [x] Make VIBE an installable package
- [ ] Fix **GRU memory discontinuous between batches** in demo.py
- [ ] Add **realtime live interface** which processes the video stream frame-by-frame

# Explain
1. #### Pip installable. 

- This repository put the original project into "vibe" folder, correct corresponding imports, and add some `__init__.py` files. Now it can be installed with:
```
pip install git+https://github.com/zc402/VIBE.git
```

2. #### GRU memory reset problem: (fix under development)

- The original vibe.py **reset** GRU memory between batches, which causes discontinuous predictions. This repo tries to fix that.

- This default behavior is probably just for simplicity. keeping hidden states for multi-person is troublesome. 

- The GRU hidden state is `reset` at:
```
# /home/zc/Projects/VIBE/vibe/lib/models/vibe.py
# class TemporalEncoder
# def forward()
y, _ = self.gru(x)

# The "_" is the final hidden state and should be preserved
# https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
```

- This repo preserve GRU hidden state within the **lifecycle of the model**, instead of one batch.

3. #### Realtime interface (Under development)

- This feature will hopefully make VIBE runnable on webcam.

- Processing steps of the original VIBE :
  - use ffmpeg to **split video** into images, save to /tmp 
  - process the human **tracking** for whole video, keep results in memory
  - predict smpl params with VIBE for whole video, 1 person at a time.
  - (optional) render and show (frame by frame)
  - save rendered result

- Processing steps of realtime live interface
  - create VIBE model.
  - read a frame with cv2
  - run tracking for 1 frame
  - predict smpl params for each person, keep the hidden states separately.
  - (optional) render and show