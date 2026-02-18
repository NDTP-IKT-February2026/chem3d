from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import io
import zipfile
import os

from model_generator import generate_3d_molecule

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # В продакшне замените на конкретные origins
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/generate_chemistry_3d/')
async def generate_chemistry_3d(chemistry_formule: str):
    result = generate_3d_molecule(chemistry_formule)
    if 'error' in result:
        raise HTTPException(status_code=422, detail='no chemistry molecule')
    
    obj = result['obj']
    mtl = result['mtl']
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.writestr('model.obj', obj)
        zip_file.writestr('model.mtl', mtl)
        
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer, 
        media_type='application/x-zip-compressed',
        headers={'Content-Disposition': 'attachment; filename=model.zip'}
    )

@app.get('/atom_model/')
async def atom_model(atom_name: str):
    atom_model_path = os.path.join('atom_models', f'{atom_name}.zip')
    if not os.path.exists(atom_model_path):
        raise HTTPException(status_code=404, detail='no atom model')
    return FileResponse(path=f'atom_models\\{atom_name}.zip', filename='model.zip', media_type='application/x-zip-compressed')