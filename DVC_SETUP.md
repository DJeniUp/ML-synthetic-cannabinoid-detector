# DVC Setup — Google Drive Remote

This project uses [DVC](https://dvc.org) to version data and model artifacts.
The remote storage is Google Drive.

---

## 1. Create the Google Drive folder (do this once, manually in browser)

1. Open [Google Drive](https://drive.google.com) and sign in.
2. Click **New → Folder**, name it e.g. `cb1-dvc-storage`.
3. Open the folder. Look at the browser URL — it will look like:
   ```
   https://drive.google.com/drive/folders/1aBcDeFgHiJkLmNoPqRsTuVwXyZ012345
   ```
   Copy the long string at the end — that is your **Folder ID**.

---

## 2. Share the folder with the course instructor

1. Right-click the folder → **Share**.
2. Add `aaaksenova2@gmail.com` with **Editor** access.
3. Click **Send**.

---

## 3. Paste your Folder ID into the DVC config

Run this command, replacing `<YOUR_FOLDER_ID>` with the ID you copied in step 1:

```bash
uv run dvc remote modify storage url gdrive://<YOUR_FOLDER_ID>
```

Verify the config looks correct:
```bash
cat .dvc/config
# Should show:
# [core]
#     remote = storage
# ['remote "storage"']
#     url = gdrive://1aBcDeFgHiJkLmNoPqRsTuVwXyZ012345
```

---

## 4. Authenticate with Google and push artifacts

DVC uses OAuth to authenticate. Run:

```bash
uv run dvc push
```

A browser window will open asking you to log into Google and grant DVC access.
After you approve, the push will resume automatically.

Artifacts pushed:
- `data/cb1_raw.csv`
- `data/train.csv`
- `data/test.csv`
- `modeling/models/xgb_model.json`

---

## 5. Commit the DVC pointer files to git

```bash
git add .dvc/config data/cb1_raw.csv.dvc data/train.csv.dvc data/test.csv.dvc \
        modeling/models/xgb_model.json.dvc \
        data/.gitignore modeling/models/.gitignore \
        .gitignore DVC_SETUP.md pyproject.toml uv.lock
git commit -m "Add DVC tracking with Google Drive remote"
git push
```

---

## 6. Reproducing from scratch (for a colleague)

```bash
# 1. Clone the repo
git clone <repo-url>
cd synthetic-cannabinoid-detector

# 2. Install DVC (at repo root)
uv sync

# 3. Pull data and model artifacts from Google Drive
#    (will open a browser for Google OAuth on first run)
uv run dvc pull

# 4. Train (optional — model already pulled)
cd modeling
uv sync
uv run python train.py

# 5. Serve
cd ../serving
uv sync
cp ../modeling/models/xgb_model.json models/
uv run uvicorn app.main:app --reload
```

---

## Reference: DVC pointer files in this repo

| DVC file | Tracks |
|----------|--------|
| `data/cb1_raw.csv.dvc` | Raw ChEMBL data (8 589 rows) |
| `data/train.csv.dvc` | Temporal train split (year ≤ 2017, 4 798 molecules) |
| `data/test.csv.dvc` | Temporal test split (year > 2017, 993 molecules) |
| `modeling/models/xgb_model.json.dvc` | Trained XGBoost model |

These `.dvc` files are small JSON-like pointers (md5 hash + size) committed to git.
The actual file contents live in Google Drive and are never committed to git.
