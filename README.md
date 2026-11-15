# Link Budget Calculator

Desktop application (Python + NiceGUI) for microwave link budget / propagation calculations
(ITU-R method, terrain horizon, land/sea path composition, maps).

There are two ways to use it:



## A. Using the ready-made .exe

Access this link: 
<br>https://adauniversity-my.sharepoint.com/:f:/g/personal/falakbarli20508_ada_edu_az/IgDqEzm_156UTpXL2f5wGuEOAbdObSq-qvW2-cJ8Jk8UD8k?e=dZ17B5


1. Unzip the **whole** folder, then double-click `LinkBudgetCalculator.exe` inside it.
2. Windows may show "Windows protected your PC" (the exe is not code-signed): click
   **More info -> Run anyway**.
3. If the window stays blank, install the free *Microsoft Edge WebView2 Runtime* and start again.

---

## B. Running from source

### 1. What you need
- Windows 10/11, 64-bit
- **Python 3.12 (64-bit)** from https://www.python.org/downloads/ - tick **"Add python.exe to PATH"**
  during installation. (Newer versions such as 3.13+ may not have wheels for every library yet.)
- Internet connection for the one-time setup
- Microsoft Edge WebView2 Runtime (already present on Windows 11 and most Windows 10 PCs)
- Git (optional - you can also use *Code -> Download ZIP* on GitHub)

### 2. Get the code
```powershell
git clone <REPOSITORY-URL>
cd <REPOSITORY-FOLDER>
```

### 3. Get the data files  (`data_sources/`)
The program needs these files inside the `data_sources` folder (next to `desktop_app.py`):

| File | Purpose |
|---|---|
| `azerbaijan_dem.tif` | elevation model (terrain profile, map) |
| `horizon_grid.db` | pre-computed terrain horizon grid |
| `ne_10m_admin_0_countries.geojson` (or `.shp` set) | country borders / land-sea |
| `ne_10m_coastline/ne_10m_coastline.shp` (+ `.shx .dbf .prj .cpg`) | coastline distance |
| `dem_hillshade.png`, `dem_bounds.json` | map relief overlay (generated in step 6) |

Access data_sources from here:
<br> https://adauniversity-my.sharepoint.com/:f:/g/personal/falakbarli20508_ada_edu_az/IgD888f7ZCTqRqmHi3Rv_QpPAY934Oev9YHPFeEVl_VUFaQ?e=gFUvR5


### 4. Create a virtual environment and install the libraries
Run these in **PowerShell** inside the project folder:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell says *"running scripts is disabled"*, run this once and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
(or use `cmd` and `venv\Scripts\activate.bat` instead).

### 5. Offline map files (only if `ui/static/web_assets/` is missing or incomplete)
```powershell
python tools/fetch_web_assets.py
```
Downloads Leaflet / Bootstrap / jQuery / Font Awesome once so maps also work offline.
If **one file fails with `CERTIFICATE_VERIFY_FAILED`** (typically jQuery), download it by hand:
```powershell
New-Item -ItemType Directory -Force ui\static\web_assets\jquery
Invoke-WebRequest https://cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.min.js -OutFile ui\static\web_assets\jquery\jquery.min.js
```

### 6. Relief overlay (only if `dem_hillshade.png` / `dem_bounds.json` are missing)
```powershell
python tools/prepare_dem_overlay.py
```

### 7. Start the program
```powershell
python desktop_app.py
```
A desktop window "Link Budget Calculator" opens. Next time you only need:
```powershell
.\venv\Scripts\Activate.ps1
python desktop_app.py
```

---

## Where does the program save things?
- Autosave and logs: `%LOCALAPPDATA%\LinkBudgetCalculator\`
  (`autosave.json`, `logs\app.log`) - paste the end of `app.log` when reporting a problem.
- Projects: wherever you choose in the *Save project* dialog.

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: ...` | venv not activated, or `pip install -r requirements.txt` not finished |
| `pip` fails building `rasterio`/`pyproj` | use 64-bit **Python 3.12**; run `python -m pip install --upgrade pip` |
| Window is blank / does not open | install Microsoft Edge WebView2 Runtime |
| Map has no background relief | run step 6 |
| Map controls/icons missing offline | run step 5 |
| "data file not found" | check step 3 - folder must be named exactly `data_sources` |
| Border distances ask for manual entry | `ne_10m_admin_0_countries.geojson` missing (otherwise it is downloaded online) |

## Building your own .exe (optional)
```powershell
pip install pyinstaller
python build_exe.py
```
Output: `dist\LinkBudgetCalculator\` and `dist\LinkBudgetCalculator_share.zip`.
Run `python build_exe.py --dry-run` first to check that nothing is missing.

## Project structure
```
desktop_app.py      entry point (NiceGUI native window)
core/               calculations, ITU import, map, terrain, path composition
ui/                 left / middle / right panes and wizard steps
tools/              fetch_web_assets.py, prepare_dem_overlay.py
data_sources/       DEM, horizon DB, borders, coastline (data, not code)
requirements.txt    Python libraries
build_exe.py        builds the Windows .exe
```

## Pin versions (for the author, once)
After the app works in your venv, record the exact versions so others get the same ones:
```powershell
pip freeze | findstr /i /b "nicegui== pywebview== folium== geopandas== shapely== pyproj== rasterio== numpy== pandas== scipy== matplotlib== requests== pyshp== access-parser== pyodbc=="
```
Copy that output into `requirements.txt`, replacing the unpinned lines.