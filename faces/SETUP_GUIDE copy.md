# Face Recognition App — Setup Guide

Follow these steps in order. This will get you from a blank PC to a running
face recognition web app.

---

## 1. Install Anaconda / Miniconda (skip if already installed)

Download and install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

Choose the Windows installer, run it, and accept the default options.

To confirm it installed correctly, open **Anaconda Prompt** (search for it in
the Start menu) and run:
```bash
conda --version
```
You should see a version number printed.

---

## 2. Get the project files

Either clone the repo:
```bash
git clone https://github.com/Ankit-dads/Deep-Learning.git
cd Deep-Learning
```
or, if you were sent a zip, extract it and open a terminal inside that folder.

Make sure `app.py` and `requirements.txt` are both present in this folder:
```bash
dir
```

---

## 3. Create the virtual environment

```bash
conda create -n face_cv python=3.10 -y
```

Activate it (do this every time before running the app, in every new terminal):
```bash
conda activate face_cv
```

You should see `(face_cv)` appear at the start of your terminal prompt.

---

## 4. Install GPU-enabled PyTorch (optional but recommended if you have an NVIDIA GPU)

Skip this step if you don't have an NVIDIA GPU or aren't sure — the app will
still work fine on CPU, just slower during training.

If you do have an NVIDIA GPU, install PyTorch with CUDA support **before**
installing the rest of the requirements:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 5. Install all remaining packages (single command)

```bash
pip install -r requirements.txt
```

This installs everything: OpenCV, NumPy, Streamlit, streamlit-webrtc, PyTorch
(if not already installed in Step 4), Pandas, av, and Pillow.

Wait for it to finish — this can take a few minutes the first time.

---

## 6. Verify the install worked

Run these two checks:
```bash
python -c "import cv2; print('OpenCV OK:', cv2.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
- First line should print an OpenCV version number.
- Second line prints `True` if GPU support is active, `False` if running on CPU
  (both are fine — `False` just means training will be slower).

---

## 7. Run the app

```bash
streamlit run app.py
```

Your browser should open automatically. If it doesn't, copy the **Local URL**
shown in the terminal (it will look like `http://localhost:8501`) and paste it
into your browser manually.

**Important:** always use the `localhost` URL, not the `Network URL` — browsers
block webcam access on non-secure (non-localhost) addresses.

---

## 8. Using the app

The app has 4 tabs:

1. **Register New Face** — type your name, click "Start Capturing," let it
   collect ~150 face images of you.
2. **Train Model** — click "Start Training" once you (and anyone else) have
   registered. Watch the live accuracy update.
3. **Live Recognition** — see yourself recognized in real time with a
   confidence score.
4. **Attendance Dashboard** — view logged recognition history and charts.

---

## Troubleshooting

**"navigator.mediaDevices is undefined" error on webcam tabs**
→ You're on the wrong URL. Switch to `http://localhost:8501`, not the network
IP address.

**Webcam permission popup never appears / camera doesn't turn on**
→ Click the padlock/site info icon in your browser's address bar → make sure
Camera permission is set to "Allow" for this site, then refresh the page.

**`pip install` fails partway through**
→ Re-run `pip install -r requirements.txt` again — pip skips anything already
installed and retries the rest. If a specific package keeps failing, note the
error message and ask for help with that specific package.

**Training tab says "disabled" / won't let you click Start Training**
→ You need at least 2 registered people in `dataset/` before training is
allowed the model needs more than one class to classify.

**Every subsequent time you want to run the app** (after this one-time setup):
```bash
conda activate face_cv
streamlit run app.py
```
That's it — no need to repeat the install steps again.
