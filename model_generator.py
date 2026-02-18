import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import re
import tempfile
from typing import Dict, Tuple, Optional, List
import io
import json

# ------------------------------
# 1. Загрузка и трансформация модели (OBJ -> массив вершин)
# ------------------------------
def load_obj_model(obj_path):
    """
    Загружает OBJ-модель, возвращает вершины, нормали и размеры.
    
    Args:
        obj_path: путь к OBJ файлу
    
    Returns:
        dict: словарь с ключами 'vertices', 'normals', 'size', 'center'
    """
    vertices = []
    normals = []
    faces = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            if parts[0] == 'v':
                # Вершина
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vn':
                # Нормаль
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # Грань
                faces.append(parts[1:])
    
    vertices = np.array(vertices)
    normals = np.array(normals) if normals else None
    
    # Если нет нормалей, генерируем их
    if normals is None or len(normals) == 0:
        normals = generate_normals(vertices, faces)
    
    # Создаем плоский список вершин и нормалей для каждой грани (как в STL)
    vertices_flat = []
    normals_flat = []
    
    for face in faces:
        for vertex_info in face:
            # Разбираем формат "v/vt/vn" или "v//vn" или "v"
            parts = vertex_info.split('/')
            v_idx = int(parts[0]) - 1  # OBJ индексы с 1
            
            if v_idx < len(vertices):
                vertices_flat.append(vertices[v_idx])
                
                # Пытаемся получить нормаль
                if len(parts) >= 3 and parts[2] and int(parts[2]) - 1 < len(normals):
                    n_idx = int(parts[2]) - 1
                    normals_flat.append(normals[n_idx])
                else:
                    # Если нормали нет, генерируем приблизительную
                    normals_flat.append([0, 1, 0])  # временно
    
    vertices_flat = np.array(vertices_flat)
    normals_flat = np.array(normals_flat)
    
    # Нормализуем нормали
    for i in range(len(normals_flat)):
        norm = np.linalg.norm(normals_flat[i])
        if norm > 0:
            normals_flat[i] = normals_flat[i] / norm
    
    min_bound = vertices_flat.min(axis=0)
    max_bound = vertices_flat.max(axis=0)
    size = max_bound - min_bound
    center = (min_bound + max_bound) / 2
    
    return {
        'vertices': vertices_flat,
        'normals': normals_flat,
        'size': size,
        'center': center,
        'original_vertices': vertices,
        'original_faces': faces
    }

def generate_normals(vertices, faces):
    """
    Генерирует нормали для вершин на основе граней.
    """
    normals = np.zeros_like(vertices)
    face_normals = []
    
    # Сначала вычисляем нормали для каждой грани
    for face in faces:
        if len(face) >= 3:
            # Получаем индексы вершин
            idx = []
            for vertex_info in face:
                parts = vertex_info.split('/')
                idx.append(int(parts[0]) - 1)
            
            # Вычисляем нормаль грани
            v1 = vertices[idx[1]] - vertices[idx[0]]
            v2 = vertices[idx[2]] - vertices[idx[0]]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            face_normals.append(normal)
    
    # Усредняем нормали для вершин
    for i, face in enumerate(faces):
        if i < len(face_normals):
            for vertex_info in face:
                parts = vertex_info.split('/')
                v_idx = int(parts[0]) - 1
                normals[v_idx] += face_normals[i]
    
    # Нормализуем
    for i in range(len(normals)):
        norm = np.linalg.norm(normals[i])
        if norm > 0:
            normals[i] = normals[i] / norm
    
    return normals

def transform_vertices(vertices, normals, scale, translation, center):
    """Масштабирует и перемещает вершины."""
    vertices_centered = vertices - center
    vertices_scaled = vertices_centered * scale
    vertices_translated = vertices_scaled + translation
    return vertices_translated, normals

# ------------------------------
# 2. Радиусы и цвета атомов
# ------------------------------
def get_atom_radius(element):
    """Ван-дер-ваальсов радиус в ангстремах."""
    radii = {
        'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47,
        'P': 1.8, 'S': 1.8, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
        'B': 1.92, 'Si': 2.1, 'Fe': 2.0, 'Mg': 1.73, 'Ca': 2.0,
    }
    return radii.get(element, 1.5)

def get_atom_color(element):
    """CPK-цвет в формате RGB (0-1 для OBJ)."""
    colors = {
        'H': (1.0, 1.0, 1.0),      # белый
        'C': (0.3, 0.3, 0.3),      # серый
        'N': (0.0, 0.0, 1.0),      # синий
        'O': (1.0, 0.0, 0.0),      # красный
        'F': (0.0, 1.0, 0.0),      # зелёный
        'Cl': (0.0, 1.0, 0.0),     # зелёный
        'Br': (0.55, 0.27, 0.07),  # коричневый
        'I': (0.58, 0.0, 0.83),    # фиолетовый
        'P': (1.0, 0.65, 0.0),     # оранжевый
        'S': (1.0, 1.0, 0.0),      # жёлтый
        'B': (1.0, 0.65, 0.0),     # оранжевый
        'Si': (0.5, 0.5, 0.5),     # серый
        'Fe': (1.0, 0.55, 0.0),    # оранжевый
        'Mg': (0.13, 0.55, 0.13),  # зелёный
        'Ca': (0.5, 0.5, 0.5),     # серый
    }
    return colors.get(element, (0.8, 0.8, 0.8))

def get_bond_color(bond_type):
    """Цвет для связей."""
    return (0.8, 0.8, 0.8)  # Светло-серый для всех связей

def get_bond_thickness(bond_type):
    """Толщина связи в зависимости от типа."""
    thickness = {
        Chem.rdchem.BondType.SINGLE: 0.2,
        Chem.rdchem.BondType.DOUBLE: 0.3,
        Chem.rdchem.BondType.TRIPLE: 0.4,
        Chem.rdchem.BondType.AROMATIC: 0.25,
    }
    return thickness.get(bond_type, 0.2)

# ------------------------------
# 3. Генерация цилиндров для связей
# ------------------------------
def create_cylinder_mesh_data(radius=0.2, height=1.0, segments=8):
    """
    Создаёт данные цилиндра для использования в качестве связи.
    
    Returns:
        dict: словарь с ключами 'vertices', 'normals', 'size', 'center'
    """
    vertices = []
    normals = []
    
    # Вершины для верхнего и нижнего кругов
    for y in [-height/2, height/2]:
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            
            vertices.append([x, y, z])
            
            # Нормаль для боковой поверхности
            norm = np.array([x, 0, z])
            norm = norm / np.linalg.norm(norm)
            normals.append(norm)
    
    # Создаем треугольники для боковой поверхности
    faces_vertices = []
    for i in range(segments):
        next_i = (i + 1) % segments
        
        # Треугольники для боковой поверхности
        p1 = i
        p2 = next_i
        p3 = i + segments
        p4 = next_i + segments
        
        faces_vertices.append([p1, p2, p4])
        faces_vertices.append([p1, p4, p3])
    
    # Преобразуем в формат плоского списка вершин
    vertices_flat = []
    normals_flat = []
    
    for face in faces_vertices:
        for idx in face:
            vertices_flat.append(vertices[idx])
            normals_flat.append(normals[idx])
    
    vertices_flat = np.array(vertices_flat)
    normals_flat = np.array(normals_flat)
    
    min_bound = vertices_flat.min(axis=0)
    max_bound = vertices_flat.max(axis=0)
    size = max_bound - min_bound
    center = (min_bound + max_bound) / 2
    
    return {
        'vertices': vertices_flat,
        'normals': normals_flat,
        'size': size,
        'center': center
    }

def create_bond_cylinder(pos1, pos2, bond_type, cylinder_model):
    """
    Создаёт цилиндр для связи между двумя атомами.
    
    Args:
        pos1, pos2: позиции атомов
        bond_type: тип связи
        cylinder_model: базовая модель цилиндра
    
    Returns:
        dict: данные для связи
    """
    p1 = np.array(pos1)
    p2 = np.array(pos2)
    
    # Вектор связи
    bond_vector = p2 - p1
    bond_length = np.linalg.norm(bond_vector)
    
    if bond_length < 0.1:
        return None
    
    # Направление связи
    direction = bond_vector / bond_length
    
    # Базовая ориентация цилиндра (вдоль Y)
    base_dir = np.array([0, 1, 0])
    
    # Вычисляем матрицу поворота
    if np.allclose(direction, base_dir):
        rotation_matrix = np.eye(3)
    elif np.allclose(direction, -base_dir):
        rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        v = np.cross(base_dir, direction)
        s = np.linalg.norm(v)
        c = np.dot(base_dir, direction)
        
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        
        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    
    # Масштабируем по высоте
    scale_matrix = np.diag([1.0, bond_length, 1.0])
    
    # Применяем преобразования к вершинам
    vertices = cylinder_model['vertices'].copy()
    
    # Сначала масштабируем, потом поворачиваем
    vertices = np.dot(vertices, scale_matrix.T)
    vertices = np.dot(vertices, rotation_matrix.T)
    
    # Перемещаем в центр связи
    center_pos = (p1 + p2) / 2
    vertices = vertices + center_pos
    
    # Поворачиваем нормали
    normals = cylinder_model['normals'].copy()
    normals = np.dot(normals, rotation_matrix.T)
    
    # Нормализуем нормали
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms
    
    return {
        'vertices': vertices,
        'normals': normals,
        'bond_type': bond_type,
        'position': center_pos
    }

# ------------------------------
# 4. Генерация OBJ с моделями и цветами
# ------------------------------
def generate_obj_strings(atom_instances, bond_instances=None):
    """
    Создаёт содержимое OBJ и MTL файлов в виде строк.
    
    Args:
        atom_instances: список словарей с данными атомов
        bond_instances: список словарей с данными связей (опционально)
        
    Returns:
        tuple: (obj_content, mtl_content) - строки с содержимым файлов
    """
    obj_lines = []
    mtl_lines = []
    
    # Заголовок OBJ
    obj_lines.append("# OBJ file for molecule")
    obj_lines.append("mtllib model.mtl")
    obj_lines.append("o molecule")
    obj_lines.append("")
    
    # Добавляем комментарии с информацией об атомах в начале файла
    obj_lines.append("# ATOM_INFO: index,element,position_x,position_y,position_z")
    for i, atom in enumerate(atom_instances):
        pos = atom['position']
        obj_lines.append(f"# ATOM_{i}: {atom['element']} {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
    obj_lines.append("")
    
    materials_created = set()
    vertex_offset = 1
    
    # Добавляем атомы
    for i, atom in enumerate(atom_instances):
        element = atom['element']
        color = atom['color']
        vertices = atom['vertices']
        normals = atom['normals']
        
        # Создаем отдельную группу для каждого атома с уникальным именем
        group_name = f"atom_{element}_{i}"
        obj_lines.append(f"g {group_name}")
        obj_lines.append(f"# ATOM_DATA: {element} {i}")
        
        material_name = f"mat_{group_name}"
        
        if material_name not in materials_created:
            mtl_lines.append(f"newmtl {material_name}")
            mtl_lines.append(f"Ka {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}")
            mtl_lines.append(f"Kd {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}")
            mtl_lines.append(f"Ks 0.2 0.2 0.2")
            mtl_lines.append(f"Ns 50")
            mtl_lines.append(f"d 1.0")
            mtl_lines.append(f"Illum 2")
            mtl_lines.append("")
            materials_created.add(material_name)
        
        obj_lines.append(f"usemtl {material_name}")
        
        # Вершины
        vertex_start = vertex_offset
        for v in vertices:
            obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        
        # Нормали
        for n in normals:
            obj_lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
        
        # Фейсы (треугольники)
        num_vertices = len(vertices)
        for j in range(0, num_vertices, 3):
            v1 = vertex_start + j
            v2 = vertex_start + j + 1
            v3 = vertex_start + j + 2
            obj_lines.append(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}")
        
        vertex_offset += num_vertices
        obj_lines.append("")
    
    # Добавляем связи, если они есть
    if bond_instances:
        for i, bond in enumerate(bond_instances):
            if bond is None:
                continue
                
            vertices = bond['vertices']
            normals = bond['normals']
            bond_type = bond.get('bond_type', Chem.rdchem.BondType.SINGLE)
            
            # Используем единый материал для всех связей
            material_name = f"bond_material"
            bond_color = get_bond_color(bond_type)
            
            if material_name not in materials_created:
                mtl_lines.append(f"newmtl {material_name}")
                mtl_lines.append(f"Ka {bond_color[0]:.3f} {bond_color[1]:.3f} {bond_color[2]:.3f}")
                mtl_lines.append(f"Kd {bond_color[0]:.3f} {bond_color[1]:.3f} {bond_color[2]:.3f}")
                mtl_lines.append(f"Ks 0.3 0.3 0.3")
                mtl_lines.append(f"Ns 30")
                mtl_lines.append(f"d 1.0")
                mtl_lines.append(f"Illum 2")
                mtl_lines.append("")
                materials_created.add(material_name)
            
            obj_lines.append(f"# Bond {i}")
            obj_lines.append(f"g bond_{i}")
            obj_lines.append(f"usemtl {material_name}")
            
            # Вершины
            for v in vertices:
                obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
            
            # Нормали
            for n in normals:
                obj_lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
            
            # Фейсы
            num_vertices = len(vertices)
            for j in range(0, num_vertices, 3):
                v1 = vertex_offset + j
                v2 = vertex_offset + j + 1
                v3 = vertex_offset + j + 2
                obj_lines.append(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}")
            
            vertex_offset += num_vertices
            obj_lines.append("")
    
    return "\n".join(obj_lines), "\n".join(mtl_lines)

def generate_obj_from_mol_with_models(mol, atom_model_data, cylinder_model_data):
    """
    Создаёт OBJ-строки: на месте каждого атома – ваша модель, и связи между ними.
    
    Args:
        mol: RDKit молекула
        atom_model_data: словарь с данными модели атома (из OBJ)
        cylinder_model_data: словарь с данными модели цилиндра для связей
    
    Returns:
        tuple: (obj_content, mtl_content)
    """
    conf = mol.GetConformer()
    atom_instances = []
    atom_positions = {}
    
    # Определяем размер модели атома для масштабирования
    model_size = max(atom_model_data['size'])
    model_radius = model_size / 0.5
    
    # Создаем атомы
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        element = atom.GetSymbol()
        
        desired_radius = get_atom_radius(element)
        scale = desired_radius / model_radius
        translation = np.array([pos.x, pos.y, pos.z])
        
        vertices_transformed, normals = transform_vertices(
            atom_model_data['vertices'],
            atom_model_data['normals'],
            scale,
            translation,
            atom_model_data['center']
        )
        
        atom_instances.append({
            'element': element,
            'vertices': vertices_transformed,
            'normals': normals,
            'color': get_atom_color(element),
            'position': translation
        })
        
        atom_positions[idx] = np.array([pos.x, pos.y, pos.z])
    
    # Создаем связи
    bond_instances = []
    
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        
        pos1 = atom_positions[begin_idx]
        pos2 = atom_positions[end_idx]
        
        # Для кратных связей создаем несколько цилиндров со смещением
        if bond_type == Chem.rdchem.BondType.DOUBLE:
            # Создаем две параллельные связи
            offset = 0.15
            direction = pos2 - pos1
            perp = np.cross(direction, np.array([0, 1, 0]))
            if np.linalg.norm(perp) < 0.1:
                perp = np.cross(direction, np.array([1, 0, 0]))
            perp = perp / np.linalg.norm(perp) * offset
            
            # Первая связь со смещением в одну сторону
            bond1 = create_bond_cylinder(pos1 + perp, pos2 + perp, bond_type, cylinder_model_data)
            # Вторая связь со смещением в другую сторону
            bond2 = create_bond_cylinder(pos1 - perp, pos2 - perp, bond_type, cylinder_model_data)
            
            if bond1:
                bond_instances.append(bond1)
            if bond2:
                bond_instances.append(bond2)
                
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            # Создаем три связи: одна по центру, две по бокам
            offset = 0.2
            direction = pos2 - pos1
            perp1 = np.cross(direction, np.array([0, 1, 0]))
            if np.linalg.norm(perp1) < 0.1:
                perp1 = np.cross(direction, np.array([1, 0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            
            # Второе перпендикулярное направление
            perp2 = np.cross(direction, perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            # Три связи
            bond1 = create_bond_cylinder(pos1 + perp1 * offset, pos2 + perp1 * offset, bond_type, cylinder_model_data)
            bond2 = create_bond_cylinder(pos1 - perp1 * offset, pos2 - perp1 * offset, bond_type, cylinder_model_data)
            bond3 = create_bond_cylinder(pos1, pos2, bond_type, cylinder_model_data)
            
            if bond1:
                bond_instances.append(bond1)
            if bond2:
                bond_instances.append(bond2)
            if bond3:
                bond_instances.append(bond3)
        else:
            # Одинарная или ароматическая связь
            bond = create_bond_cylinder(pos1, pos2, bond_type, cylinder_model_data)
            if bond:
                bond_instances.append(bond)
    
    return generate_obj_strings(atom_instances, bond_instances)

# ------------------------------
# 5. Формула -> 3D
# ------------------------------
def formula_to_3d(formula):
    """Преобразование химической формулы в 3D-структуру (RDKit mol)"""
    formula_db = {
        "H2O": "O", 
        "CO2": "O=C=O", 
        "CH4": "C", 
        "NH3": "N",
        "O2": "O=O", 
        "N2": "N#N", 
        "C2H6": "CC", 
        "C2H4": "C=C",
        "C2H2": "C#C", 
        "C3H8": "CCC", 
        "C4H10": "CCCC",
        "C6H6": "c1ccccc1", 
        "C2H5OH": "CCO", 
        "CH3COOH": "CC(=O)O",
        "C6H12O6": "C(C(C(C(C(C=O)O)O)O)O)O",
        "C9H8O4": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "C8H9NO2": "CC(=O)Nc1ccc(O)cc1",
        "C3H6O": "CC(=O)C",
        "CH3OH": "CO",
    }

    if formula in formula_db:
        smiles = formula_db[formula]
    else:
        # Пытаемся создать простую цепочку
        smiles = create_simple_smiles(formula)
        if not smiles:
            return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if bool(re.search(r'H\d*', formula)):
        mol = Chem.AddHs(mol)
    
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    
    return mol

def create_simple_smiles(formula):
    """Создание простой цепочки из тяжёлых атомов."""
    elements = []
    for elem, count in re.findall(r'([A-Z][a-z]*)(\d*)', formula):
        count = int(count) if count else 1
        elements.extend([elem] * count)

    heavy_atoms = [e for e in elements if e != 'H']
    if not heavy_atoms:
        return None

    smiles = ""
    for atom in heavy_atoms:
        if atom == 'C':
            smiles += 'C'
        elif atom == 'O':
            smiles += 'O'
        elif atom == 'N':
            smiles += 'N'
        else:
            smiles += f'[{atom}]'
    return smiles

# ------------------------------
# 6. Создание тестовой модели сферы в формате OBJ
# ------------------------------
def create_sphere_obj_data(radius=1.0, segments=32):
    """
    Создаёт данные сферы в формате OBJ.
    
    Returns:
        dict: словарь с данными модели
    """
    vertices = []
    normals = []
    faces = []
    
    # Создаем вершины сферы
    for i in range(segments + 1):
        theta = i * np.pi / segments
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        for j in range(segments + 1):
            phi = j * 2 * np.pi / segments
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            
            x = radius * sin_theta * cos_phi
            y = radius * sin_theta * sin_phi
            z = radius * cos_theta
            
            vertices.append([x, y, z])
            
            # Нормаль для сферы
            norm = [x, y, z]
            norm = norm / np.linalg.norm(norm)
            normals.append(norm)
    
    # Создаем грани
    for i in range(segments):
        for j in range(segments):
            p1 = i * (segments + 1) + j
            p2 = p1 + 1
            p3 = (i + 1) * (segments + 1) + j
            p4 = p3 + 1
            
            # Каждая грань - треугольник (в OBJ индексы с 1)
            faces.append([p1 + 1, p2 + 1, p3 + 1])
            faces.append([p2 + 1, p4 + 1, p3 + 1])
    
    # Создаем плоский список вершин для совместимости
    vertices_flat = []
    normals_flat = []
    
    for face in faces:
        for idx in face:
            vertices_flat.append(vertices[idx - 1])
            normals_flat.append(normals[idx - 1])
    
    vertices_flat = np.array(vertices_flat)
    normals_flat = np.array(normals_flat)
    
    min_bound = vertices_flat.min(axis=0)
    max_bound = vertices_flat.max(axis=0)
    size = max_bound - min_bound
    center = (min_bound + max_bound) / 2
    
    return {
        'vertices': vertices_flat,
        'normals': normals_flat,
        'size': size,
        'center': center,
        'original_vertices': vertices,
        'original_faces': faces
    }

# ------------------------------
# 7. Основная функция
# ------------------------------
def generate_3d_molecule(formula: str, use_sphere_model: bool = True, custom_model_path: str = None) -> Dict[str, str]:
    """
    Генерирует 3D модель молекулы по формуле с атомами и связями.
    
    Args:
        formula: Химическая формула (например, "H2O", "CH4")
        use_sphere_model: Использовать сферическую модель (если True)
        custom_model_path: Путь к пользовательской OBJ модели (если use_sphere_model=False)
    
    Returns:
        dict: Словарь с ключами 'obj' и 'mtl', содержащими строки файлов
    """
    # Получаем модель для атомов
    if use_sphere_model:
        atom_model_data = create_sphere_obj_data(radius=1.0)
    elif custom_model_path and os.path.exists(custom_model_path):
        atom_model_data = load_obj_model(custom_model_path)
    else:
        return {"error": "Не указана корректная модель для атомов"}
    
    # Создаем модель для связей (цилиндр)
    cylinder_model_data = create_cylinder_mesh_data(radius=0.15, height=1.0, segments=16)
    
    # Создаем молекулу
    mol = formula_to_3d(formula)
    if mol is None:
        return {"error": f"Не удалось создать молекулу для формулы {formula}"}
    
    try:
        obj_content, mtl_content = generate_obj_from_mol_with_models(mol, atom_model_data, cylinder_model_data)
        
        # Формируем результат
        result = {
            'obj': obj_content,
            'mtl': mtl_content
        }
        
        # Добавляем информацию о количестве атомов и связей
        result['atom_count'] = str(mol.GetNumAtoms())
        result['bond_count'] = str(mol.GetNumBonds())
        result['formula'] = formula
        
        return result
    except Exception as e:
        return {"error": f"Ошибка при генерации: {str(e)}"}

# ------------------------------
# 8. Утилита для загрузки пользовательской OBJ модели
# ------------------------------
def load_obj_model_from_bytes(obj_bytes: bytes):
    """
    Загружает OBJ модель из байтов.
    
    Args:
        obj_bytes: байты OBJ файла
    
    Returns:
        dict: данные модели или None при ошибке
    """
    try:
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix='.obj', mode='wb', delete=False) as tmp:
            tmp.write(obj_bytes)
            tmp_path = tmp.name
        
        # Загружаем модель
        model_data = load_obj_model(tmp_path)
        
        # Удаляем временный файл
        os.unlink(tmp_path)
        
        return model_data
    except Exception as e:
        print(f"Ошибка загрузки OBJ: {e}")
        return None

# ------------------------------
# Пример использования
# ------------------------------
if __name__ == "__main__":
    print("🚀 Генератор 3D молекул с атомами и связями (OBJ формат)")
    print("=" * 70)
    
    # Тестируем на разных молекулах
    molecules = ["H2O", "CH4", "CO2", "NH3", "C2H4", "C2H2"]
    
    for formula in molecules:
        print(f"\n🔬 Генерация {formula}...")
        result = generate_3d_molecule(formula, use_sphere_model=True)
        
        if "error" in result:
            print(f"   ❌ {result['error']}")
        else:
            obj_content = result['obj']
            mtl_content = result['mtl']
            
            print(f"   ✅ Успешно!")
            print(f"   📊 Атомов: {result['atom_count']}")
            print(f"   🔗 Связей: {result['bond_count']}")
            print(f"   📄 OBJ: {len(obj_content)} символов")
            print(f"   📄 MTL: {len(mtl_content)} символов")
            
            # Показываем первые несколько строк
            print("\n   Первые 5 строк OBJ:")
            for line in obj_content.split('\n')[:5]:
                print(f"     {line}")
    
    print("\n✨ Готово!")