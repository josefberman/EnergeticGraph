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
Navigate to: `http://localhost:5000`

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
