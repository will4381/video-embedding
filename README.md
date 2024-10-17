# Video-Text Matching

This repository performs video-text matching using OpenAI's CLIP model. It allows you to search for specific content within a video using natural language queries.

## Installation

Install the required packages using pip:

```bash
pip install transformers ftfy opencv-python matplotlib imageio
```

## Usage

1. **Set the Video Path and Query:**

   - Update the `video_path` variable with the path to your video file:

     ```python
     video_path = "content/IMG_3181.mp4"
     ```

   - Set the `text_query` variable with your search query:

     ```python
     text_query = "golf follow through"
     ```

2. **Adjust Parameters (Optional):**

   - `clip_length`: Number of frames in each clip (default is `8`).
   - `frame_stride`: Number of frames to move the window for the next clip (default is `2`).
   - `threshold`: Similarity threshold for selecting matching clips (default is `0.3`).

3. **Run the Script:**

   - Execute the script to perform the search.

4. **View the Results:**

   - The script will generate a GIF of the top matching clips saved as `content/output.gif`.

## Sampling Algorithm

The video is processed by sampling overlapping clips to capture temporal context.

- **Algorithm:**

  For each clip starting at frame index $`( i )`$:

  - **Clip Selection:**

    $`
    \text{clip} = text{frames}[i : i + \text{clip\_length}]
    `$

  - **Increment Index:**

    $`
    i = i + \text{frame\_stride}
    `$

  Repeat until all frames are processed.

- **Parameters:**

  - $`( N )`$: Total number of frames in the video.
  - `clip_length`: Number of frames per clip.
  - `frame_stride`: Number of frames to skip for the next clip.

- **Visualization:**

  ```
  Frames:    [F0, F1, F2, F3, F4, F5, F6, F7, F8, F9, ...]
              |--------|
                 Clip 1
                  |--------|
                     Clip 2
  ```

## Embedding and Similarity Calculation

- **Embedding Extraction:**

  - **Frame Embeddings:**

    For each frame $`( k )`$ in a clip:

    $`
    E_{\text{frame}_k} = \text{CLIP\_ImageEncoder}(\text{frame}_k)
    `$

  - **Clip Embedding:**

    Average the frame embeddings:

    $`
    E_{\text{clip}} = \frac{1}{\text{clip\_length}} \sum_{k=1}^{\text{clip\_length}} E_{\text{frame}_k}
    `$

- **Text Embedding:**

  Extract the embedding of the text query:

  $`
  E_{\text{text}} = \text{CLIP\_TextEncoder}(\text{query})
  `$

- **Similarity Calculation:**

  Compute the cosine similarity between the text embedding and each clip embedding:

  $`
  \text{Similarity} = \frac{E_{\text{clip}} \cdot E_{\text{text}}}{\|E_{\text{clip}}\| \times \|E_{\text{text}}\|}
  `$

- **Thresholding:**

  Select clips where:

  $`
  \text{Similarity} \geq \text{threshold}
  `$

## Example

### Original Video

<img src="content/original.gif" alt="Original Video" width="400">

*(The video `content/IMG_3181.mp4` has been converted to `content/original.gif` for display purposes.)*

### Query

```plaintext
"golf follow through"
```

### Generated GIF

<img src="content/output.gif" alt="Generated GIF" width="400">

*(The GIF `content/output.gif` is generated after running the script.)*

---

Feel free to explore different queries and adjust parameters to improve the matching results!
