# Molecular Design System - Web GUI

## Quick Start

### 1. Install Flask
```bash
conda activate energetic_env
pip install flask
```

### 2. Run the GUI
```bash
cd gui
python app.py
```

### 3. Open in Browser
Navigate to: `http://localhost:5001` (see `app.py`).

## GitHub Pages (static UI)

The repository includes a static copy of the interface under `docs/` (`index.html`, `css/`, `js/`) for hosting on [GitHub Pages](https://pages.github.com/) (e.g. **Settings → Pages → Branch `main` / folder `/docs`**).

1. **Backend still required**: The Pages site is HTML/CSS/JS only. Run `python app.py` on your machine or a server; the browser must be able to reach it.

2. **Set the API URL** on the static page:
   - **Save & use** the “Base URL” field (e.g. `http://127.0.0.1:5001`), or
   - Open the site with a query: `?api=http://127.0.0.1:5001`, or
   - Set `localStorage` key `emgApiBase` (same as clicking Save).

3. **CORS**: For cross-origin requests (GitHub Pages → your API), set an environment variable when starting Flask:
   ```bash
   export EMG_CORS_ORIGINS='https://YOUR_USERNAME.github.io'
   # or for local experiments:
   export EMG_CORS_ORIGINS='*'
   python app.py
   ```
   Use your exact GitHub Pages origin (scheme + host, no path).

4. **Keeping `docs/` in sync**: After changing `gui/static/css/style.css` or `gui/static/js/main.js`, copy the files into `docs/css/` and `docs/js/` (or re-run your sync step) before committing.

## Features

### 🎨 **Modern Dark UI**
- Glassmorphic design with cyan/blue/purple gradients
- Responsive layout with 3-column grid
- Smooth animations and transitions

### 📊 **Real-Time Visualization**
- Live beam search progress updates
- Iteration-by-iteration candidate display
- Molecule structure images (RDKit-generated)

### 🧪 **Interactive Controls**
- Target property inputs (Density, Velocity, Pressure, Hf)
- RAG toggle switch
- Beam width and iteration controls
- One-click optimization start

### 📈 **Comprehensive Display**
- **Seed Molecule**: Initial starting point with properties
- **Target Properties**: Your desired values
- **Beam Candidates**: All candidates per iteration with thumbnails
- **Best Molecule**: Current best solution with full details
- **Progress Stats**: Iteration count, candidates, and best score

## Architecture

```
gui/
├── app.py                    # Flask backend with SSE
├── templates/
│   └── index.html            # Main UI template
├── static/
│   ├── css/
│   │   └── style.css         # Modern dark theme
│   ├── js/
│   │   └── main.js           # Real-time updates
│   └── molecules/            # Generated molecule images
```

## How It Works

1. **Input**: Enter target properties and configuration
2. **Start**: Click "Start Optimization" to begin
3. **Progress**: Watch real-time updates via Server-Sent Events
4. **Visualize**: See molecule structures and beam search candidates
5. **Results**: View the best molecule found with all properties

## Tech Stack

- **Backend**: Flask + Python
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Visualization**: RDKit (molecule images)
- **Real-time**: Server-Sent Events (SSE)
- **Theme**: Custom CSS with gradients and glassmorphism

Enjoy your modern molecular design interface! 🚀
