# Features

-   **Molecular Model Generation**: Convert chemical formulas (e.g., "H2O", "CH4", "C2H5OH") into 3D molecular structures
    
-   **Interactive Visualization**: Rotate, zoom, and explore molecules in real-time using Three.js
    
-   **Atom Selection**: Click on any atom to display detailed information (element, index, position)
    
-   **Atom Model Viewer**: Switch from molecule view to detailed 3D atom models (pre-modeled in 3ds Max)
    
-   **Atom List**: Browse all atoms in the current molecule with position data
    
-   **Smooth Rendering**: Advanced shading and smoothing for realistic visualization
    

# Technology Stack

### Backend (Python/FastAPI)

-   **FastAPI**: REST API server
    
-   **RDKit**: Molecular structure generation and manipulation
    
-   **NumPy**: Mathematical operations for 3D transformations
    
-   **Custom OBJ/MTL Generator**: Creates 3D models with atom colors, radii, and bonds
    

### Frontend (JavaScript/Three.js)

-   **Three.js**: 3D rendering engine
    
-   **OBJ/MTL Loaders**: Import 3D models
    
-   **OrbitControls**: Camera manipulation
    
-   **JSZip**: Handle zip file extraction


## Backend setup

Clone repository

### UV
#### Install uv

    pip install uv

#### Create uv project

    uv init

#### Install Python dependencies:

    uv sync
 
#### Start the FastAPI server:

    uv run uvicorn main:app --reload --port YOUR_PORT

### Venv
#### Create venv

    python -m venv .venv

#### Activate venv

Windows

    .venv\Scripts\activate.bat

Linux/MacOS

    source .venv/bin/activate


#### Install Python dependencies:

    pip install -r requirements.txt
 
#### Start the FastAPI server:

    uvicorn main:app --reload --port YOUR_PORT

### Frontend setup

The frontend is static HTML/JS/CSS. Simply open `index.html` in a browser or serve it through any web server.

## API Endpoints

### Generate Molecular Model

    GET /generate_chemistry_3d/?chemistry_formule={formula}

Returns a ZIP file containing:

-   `model.obj` - 3D geometry with atom and bond data
    
-   `model.mtl` - Material definitions with colors
    

### Get Atom Model

    GET /atom_model/?atom_name={element}

Returns a ZIP file containing the pre-modeled 3D atom model.

### Screenshots
![](https://github.com/NDTP-IKT-February2026/chem3d/blob/master/screenshots/Screenshot1.png)

![](https://github.com/NDTP-IKT-February2026/chem3d/blob/master/screenshots/Screenshot2.png)

![](https://github.com/NDTP-IKT-February2026/chem3d/blob/master/screenshots/Screenshot3.png)

![](https://github.com/NDTP-IKT-February2026/chem3d/blob/master/screenshots/Screenshot4.png)




