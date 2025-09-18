# Publish to GitHub — Step by Step

## Option A — Using Git (recommended)
1. Create a new repository on GitHub (no README/License initialized).
2. On your machine:
   ```bash
   git init -b main
   git add .
   git commit -m "Initial commit: CPU-only OpenPose hand inference with OpenCV DNN"
   git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
   git push -u origin main
   ```
3. On GitHub:
   - Open the repo page.
   - Edit `LICENSE` and replace `YOUR_NAME`.
   - Add a description in the repository Settings (see suggestions in `repo_name_description_options.json`).

## Option B — Using GitHub Web UI
1. Create a new repository on GitHub.
2. Click **Add file → Upload files** and drag-drop all repo files/folders.
3. Commit to `main`.
4. Edit `LICENSE` and repository description.

## After cloning (for users)
```bash
conda create -n openpose-hand python=3.10 -y
conda activate openpose-hand
python -m pip install --upgrade pip
pip install -r requirements.txt
python prepare_models.py --models models/hand
# put images under data/5/
python hand_infer.py --images data/5 --output outputs --models models/hand
```
