import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';

// --- Инициализация сцены, камеры, рендера ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111122);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(5, 5, 10);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = false;
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
document.body.appendChild(renderer.domElement);

// --- Рейкастинг для определения атомов ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// --- Словарь для хранения информации об атомах ---
const atomInfoMap = new Map();
// Карта для быстрого поиска по ID меша
const meshToAtomMap = new Map();

// --- Освещение (улучшенное для сглаженных моделей) ---
const ambientLight = new THREE.AmbientLight(0x404060, 1.2);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.8);
dirLight.position.set(2, 5, 3);
scene.add(dirLight);

const dirLight2 = new THREE.DirectionalLight(0xffeedd, 1.2);
dirLight2.position.set(-3, 2, 4);
scene.add(dirLight2);

const backLight = new THREE.DirectionalLight(0x4466ff, 1.0);
backLight.position.set(-3, 2, -4);
scene.add(backLight);

const pointLight1 = new THREE.PointLight(0xffaa88, 1.0);
pointLight1.position.set(3, 2, 3);
scene.add(pointLight1);

const pointLight2 = new THREE.PointLight(0x88aaff, 1.0);
pointLight2.position.set(-3, 1, 3);
scene.add(pointLight2);

const bottomLight = new THREE.PointLight(0x556688, 0.8);
bottomLight.position.set(0, -3, 0);
scene.add(bottomLight);

const topLight = new THREE.PointLight(0x88aaff, 0.5);
topLight.position.set(0, 5, 0);
scene.add(topLight);

// Вспомогательная сетка
const gridHelper = new THREE.GridHelper(20, 20, 0x8888ff, 0x333344);
gridHelper.position.y = -0.5;
scene.add(gridHelper);

// Звезды на заднем плане
const starsGeometry = new THREE.BufferGeometry();
const starsCount = 2000;
const starsPositions = new Float32Array(starsCount * 3);
for (let i = 0; i < starsCount * 3; i += 3) {
    starsPositions[i] = (Math.random() - 0.5) * 200;
    starsPositions[i+1] = (Math.random() - 0.5) * 200;
    starsPositions[i+2] = (Math.random() - 0.5) * 200;
}
starsGeometry.setAttribute('position', new THREE.BufferAttribute(starsPositions, 3));
const starsMaterial = new THREE.PointsMaterial({color: 0x88aaff, size: 0.15, transparent: true, opacity: 0.7});
const stars = new THREE.Points(starsGeometry, starsMaterial);
scene.add(stars);

// --- Управление ---
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.5;
controls.enableZoom = true;
controls.enablePan = true;
controls.target.set(0, 0.5, 0);

controls.update();

// --- Группы для моделей ---
let currentModelGroup = new THREE.Group();
let atomModelGroup = new THREE.Group();
scene.add(currentModelGroup);
scene.add(atomModelGroup);
atomModelGroup.visible = false;

// --- Элементы интерфейса ---
const inputElement = document.getElementById('formula-input');
const sendButton = document.getElementById('send-btn');
const statusDiv = document.getElementById('status');
const atomElementSpan = document.getElementById('atom-element');
const atomIndexSpan = document.getElementById('atom-index');
const atomPosXSpan = document.getElementById('atom-pos-x');
const atomPosYSpan = document.getElementById('atom-pos-y');
const atomPosZSpan = document.getElementById('atom-pos-z');
const atomListDiv = document.getElementById('atom-list');
const atomListBtn = document.getElementById('atom-list-btn');
const atomModelBtn = document.getElementById('atom-model-btn');
const backToMoleculeBtn = document.getElementById('back-to-molecule-btn');
const viewToggle = document.getElementById('view-toggle');

// Текущий выбранный атом
let currentSelectedAtom = null;
// Сохраняем оригинальную модель молекулы и данные атомов
let savedMoleculeModel = null;
let savedAtomInfoMap = null;
let savedMeshToAtomMap = null;

// --- Функция обновления статуса ---
function setStatus(message, isError = false, isLoading = false) {
    statusDiv.classList.remove('hidden');
    statusDiv.innerHTML = isLoading ? '<span class="loading-spinner"></span>' + message : message;
    statusDiv.style.color = isError ? '#ff6b6b' : '#ffd700';
    if (!isLoading && !isError) {
        setTimeout(() => {
            statusDiv.classList.add('hidden');
        }, 5000);
    }
}

// --- Функция применения сглаживания к геометрии (фиксированный уровень 3) ---
function applySmoothing(geometry) {
    // Создаем копию геометрии для изменений
    const smoothGeo = geometry.clone();
    
    // Получаем атрибуты позиции и нормали
    const positionAttribute = smoothGeo.attributes.position;
    const normalAttribute = smoothGeo.attributes.normal;
    
    if (!positionAttribute || !normalAttribute) return geometry;
    
    const positions = positionAttribute.array;
    const normals = normalAttribute.array;
    
    // Количество вершин
    const vertexCount = positions.length / 3;
    
    // Создаем карту для группировки вершин по позиции
    const vertexMap = new Map();
    
    // Функция для создания ключа позиции
    function getPositionKey(x, y, z) {
        const precision = 1000;
        return `${Math.round(x * precision)},${Math.round(y * precision)},${Math.round(z * precision)}`;
    }
    
    // Собираем все уникальные позиции и соответствующие им нормали
    for (let i = 0; i < vertexCount; i++) {
        const ix = i * 3;
        const iy = i * 3 + 1;
        const iz = i * 3 + 2;
        
        const x = positions[ix];
        const y = positions[iy];
        const z = positions[iz];
        
        const key = getPositionKey(x, y, z);
        
        if (!vertexMap.has(key)) {
            vertexMap.set(key, {
                positions: [],
                normals: [],
                indices: []
            });
        }
        
        const vertexData = vertexMap.get(key);
        vertexData.positions.push([x, y, z]);
        vertexData.normals.push([
            normals[ix] || 0,
            normals[iy] || 0,
            normals[iz] || 0
        ]);
        vertexData.indices.push(i);
    }
    
    // Вычисляем усредненные нормали для каждой уникальной позиции
    const smoothNormals = new Float32Array(positions.length);
    
    vertexMap.forEach((vertexData) => {
        const avgNormal = [0, 0, 0];
        
        // Суммируем все нормали для этой позиции
        vertexData.normals.forEach(normal => {
            avgNormal[0] += normal[0];
            avgNormal[1] += normal[1];
            avgNormal[2] += normal[2];
        });
        
        // Нормализуем
        const length = Math.sqrt(avgNormal[0] * avgNormal[0] + avgNormal[1] * avgNormal[1] + avgNormal[2] * avgNormal[2]);
        if (length > 0) {
            avgNormal[0] /= length;
            avgNormal[1] /= length;
            avgNormal[2] /= length;
        }
        
        // Применяем усредненную нормаль ко всем вершинам с этой позицией
        vertexData.indices.forEach(index => {
            const i = index * 3;
            smoothNormals[i] = avgNormal[0];
            smoothNormals[i + 1] = avgNormal[1];
            smoothNormals[i + 2] = avgNormal[2];
        });
    });
    
    // Заменяем нормали сглаженными
    smoothGeo.setAttribute('normal', new THREE.BufferAttribute(smoothNormals, 3));
    
    return smoothGeo;
}

// --- Функция применения сглаживания ко всем мешам в группе ---
function applySmoothingToGroup(group) {
    group.traverse((child) => {
        if (child.isMesh) {
            if (child.geometry) {
                // Сохраняем оригинальную геометрию если ещё не сохранили
                if (!child.userData.originalGeometry) {
                    child.userData.originalGeometry = child.geometry.clone();
                }
                
                // Применяем сглаживание
                const smoothedGeo = applySmoothing(child.userData.originalGeometry);
                child.geometry.dispose();
                child.geometry = smoothedGeo;
                
                // Обновляем материалы для лучшего отображения сглаженных поверхностей
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(mat => {
                            mat.roughness = 0.2;
                            mat.metalness = 0.1;
                            mat.emissive = new THREE.Color(0x000000);
                            mat.shininess = 60;
                            mat.flatShading = false;
                        });
                    } else {
                        child.material.roughness = 0.2;
                        child.material.metalness = 0.1;
                        child.material.emissive = new THREE.Color(0x000000);
                        child.material.shininess = 60;
                        child.material.flatShading = false;
                    }
                }
            }
        }
    });
}

// --- Функция очистки сцены от старой модели ---
function clearModel() {
    while(currentModelGroup.children.length > 0) {
        const child = currentModelGroup.children[0];
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
            if (Array.isArray(child.material)) {
                child.material.forEach(m => m.dispose());
            } else {
                child.material.dispose();
            }
        }
        currentModelGroup.remove(child);
    }
    
    // Очищаем модель атома
    while(atomModelGroup.children.length > 0) {
        const child = atomModelGroup.children[0];
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
            if (Array.isArray(child.material)) {
                child.material.forEach(m => m.dispose());
            } else {
                child.material.dispose();
            }
        }
        atomModelGroup.remove(child);
    }
    
    // Очищаем карты информации об атомах
    atomInfoMap.clear();
    meshToAtomMap.clear();
    
    // Сбрасываем информацию на панели
    atomElementSpan.textContent = '—';
    atomIndexSpan.textContent = '—';
    atomPosXSpan.textContent = '—';
    atomPosYSpan.textContent = '—';
    atomPosZSpan.textContent = '—';
    
    // Отключаем кнопку модели атома
    atomModelBtn.disabled = true;
    currentSelectedAtom = null;
    
    // Показываем модель молекулы, скрываем атом
    currentModelGroup.visible = true;
    atomModelGroup.visible = false;
    
    // Скрываем кнопки
    backToMoleculeBtn.style.display = 'none';
    viewToggle.style.display = 'none';
    
    // Сбрасываем сохраненные данные
    savedMoleculeModel = null;
    savedAtomInfoMap = null;
    savedMeshToAtomMap = null;
}

// --- Функция парсинга информации об атомах из OBJ комментариев ---
function parseAtomInfoFromObj(objText) {
    const lines = objText.split('\n');
    const atomInfo = [];
    
    lines.forEach(line => {
        const match = line.match(/# ATOM_(\d+): (\w+) ([\d.-]+) ([\d.-]+) ([\d.-]+)/);
        if (match) {
            atomInfo.push({
                index: parseInt(match[1]),
                element: match[2],
                position: new THREE.Vector3(
                    parseFloat(match[3]),
                    parseFloat(match[4]),
                    parseFloat(match[5])
                )
            });
        }
    });
    
    return atomInfo;
}

// --- Функция загрузки и отображения модели атома ---
async function loadAtomModelInScene(element) {
    setStatus(`Загрузка 3D модели атома ${element}...`, false, true);
    atomModelBtn.disabled = true;

    try {
        const baseUrl = CONFIG.ATOM_API;
        const url = new URL(baseUrl);
        url.searchParams.append('atom_name', element);
        
        console.log('Fetching atom model:', url.toString());
        
        const response = await fetch(url.toString(), {
            method: 'GET',
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const contentType = response.headers.get('content-type');
        
        let blob;
        if (contentType && contentType.includes('application/json')) {
            const jsonResponse = await response.json();
            throw new Error(jsonResponse.error || 'Не удалось получить модель атома');
        } else {
            blob = await response.blob();
        }
        
        if (blob.size === 0) {
            throw new Error('Получен пустой файл');
        }

        // Сохраняем текущую модель молекулы и данные атомов
        if (currentModelGroup.children.length > 0 && !savedMoleculeModel) {
            savedMoleculeModel = currentModelGroup.clone();
            
            savedAtomInfoMap = new Map();
            savedMeshToAtomMap = new Map();
            
            atomInfoMap.forEach((value, key) => {
                savedAtomInfoMap.set(key, value);
            });
            
            meshToAtomMap.forEach((value, key) => {
                savedMeshToAtomMap.set(key, value);
            });
        }

        // Очищаем предыдущую модель атома
        while(atomModelGroup.children.length > 0) {
            const child = atomModelGroup.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.dispose());
                } else {
                    child.material.dispose();
                }
            }
            atomModelGroup.remove(child);
        }

        // Загружаем модель атома
        await loadAtomModelFromBlob(blob, element);

        // Переключаем видимость
        currentModelGroup.visible = false;
        atomModelGroup.visible = true;
        
        // Обновляем текст кнопки
        viewToggle.style.display = 'block';
        viewToggle.textContent = '🌐 Режим: Атом ' + element;
        
        setStatus(`Модель атома ${element} загружена`, false, false);
        
    } catch (error) {
        console.error('Error loading atom model:', error);
        setStatus(`Ошибка загрузки модели: ${error.message}`, true, false);
    } finally {
        atomModelBtn.disabled = false;
    }
}

// --- Функция загрузки модели атома из Blob ---
async function loadAtomModelFromBlob(blob, element) {
    try {
        const JSZip = await import('https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js');
        
        let ZipConstructor;
        if (JSZip.default) {
            ZipConstructor = JSZip.default;
        } else if (typeof JSZip === 'function') {
            ZipConstructor = JSZip;
        } else {
            await loadScript('https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js');
            ZipConstructor = window.JSZip;
        }
        
        if (!ZipConstructor) {
            throw new Error('Не удалось загрузить JSZip');
        }
        
        const zip = new ZipConstructor();
        const loadedZip = await zip.loadAsync(blob);
        
        const objFile = Object.values(loadedZip.files).find(file => file.name.endsWith('.obj'));
        const mtlFile = Object.values(loadedZip.files).find(file => file.name.endsWith('.mtl'));
        
        if (!objFile) {
            throw new Error('В ZIP архиве не найден .obj файл');
        }

        setStatus('Загрузка модели атома...', false, true);
        
        const objText = await objFile.async('text');
        
        if (mtlFile) {
            const mtlText = await mtlFile.async('text');
            
            const mtlBlob = new Blob([mtlText], { type: 'text/plain' });
            const objBlob = new Blob([objText], { type: 'text/plain' });
            
            const mtlUrl = URL.createObjectURL(mtlBlob);
            const objUrl = URL.createObjectURL(objBlob);
            
            const mtlLoader = new MTLLoader();
            
            mtlLoader.load(mtlUrl, (materials) => {
                materials.preload();
                
                const objLoader = new OBJLoader();
                objLoader.setMaterials(materials);
                
                objLoader.load(objUrl, (object) => {
                    const processedObject = processNestedObjects(object);
                    
                    // Масштабируем модель атома
                    const box = new THREE.Box3().setFromObject(processedObject);
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const scaleFactor = maxDim > 3 ? 3 / maxDim : (maxDim < 0.5 ? 2 : 1);
                    processedObject.scale.setScalar(scaleFactor);
                    
                    // Применяем сглаживание
                    applySmoothingToGroup(processedObject);
                    
                    // Добавляем в группу атома
                    atomModelGroup.add(processedObject);
                    
                    // Центрируем камеру на модели атома
                    centerCameraOnObject(processedObject);
                    
                    URL.revokeObjectURL(mtlUrl);
                    URL.revokeObjectURL(objUrl);
                }, undefined, (error) => {
                    setStatus('Ошибка загрузки OBJ', true, false);
                    console.error(error);
                    URL.revokeObjectURL(mtlUrl);
                    URL.revokeObjectURL(objUrl);
                });
            }, undefined, (error) => {
                setStatus('Ошибка загрузки MTL', true, false);
                console.error(error);
                URL.revokeObjectURL(mtlUrl);
                URL.revokeObjectURL(objUrl);
            });
        } else {
            const objBlob = new Blob([objText], { type: 'text/plain' });
            const objUrl = URL.createObjectURL(objBlob);
            
            const objLoader = new OBJLoader();
            objLoader.load(objUrl, (object) => {
                const processedObject = processNestedObjects(object);
                
                // Масштабирование
                const box = new THREE.Box3().setFromObject(processedObject);
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const scaleFactor = maxDim > 3 ? 3 / maxDim : (maxDim < 0.5 ? 2 : 1);
                processedObject.scale.setScalar(scaleFactor);
                
                // Применяем сглаживание
                applySmoothingToGroup(processedObject);
                
                atomModelGroup.add(processedObject);
                
                centerCameraOnObject(processedObject);
                
                URL.revokeObjectURL(objUrl);
            }, undefined, (error) => {
                setStatus('Ошибка загрузки OBJ', true, false);
                console.error(error);
                URL.revokeObjectURL(objUrl);
            });
        }
    } catch (error) {
        console.error('Ошибка при работе с ZIP:', error);
        setStatus(`Ошибка распаковки ZIP: ${error.message}`, true, false);
    }
}

// --- Функция восстановления данных атомов после возврата к молекуле ---
function restoreAtomData() {
    if (!savedAtomInfoMap || !savedMeshToAtomMap) return;
    
    atomInfoMap.clear();
    meshToAtomMap.clear();
    
    savedAtomInfoMap.forEach((value, key) => {
        atomInfoMap.set(key, value);
    });
    
    savedMeshToAtomMap.forEach((value, key) => {
        meshToAtomMap.set(key, value);
    });
    
    updateAtomList();
    
    console.log('Данные атомов восстановлены:', atomInfoMap.size, 'атомов');
}

// --- Функция рекурсивной обработки вложенных объектов ---
function processNestedObjects(object) {
    object.traverse((child) => {
        if (child.isMesh) {
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(mat => {
                        mat.side = THREE.DoubleSide;
                        mat.flatShading = false;
                        mat.shininess = 60;
                        mat.needsUpdate = true;
                    });
                } else {
                    child.material.side = THREE.DoubleSide;
                    child.material.flatShading = false;
                    child.material.shininess = 60;
                    child.material.needsUpdate = true;
                }
            }
            
            if (child.geometry) {
                child.geometry.computeVertexNormals();
            }
        }
    });
    
    return object;
}

// --- Функция обновления списка атомов ---
function updateAtomList() {
    if (atomInfoMap.size === 0) {
        atomListDiv.innerHTML = '<div style="padding: 10px; color: #aaa;">Нет данных об атомах</div>';
        return;
    }
    
    let html = '';
    const atoms = Array.from(atomInfoMap.values());
    
    atoms.sort((a, b) => a.index - b.index);
    
    atoms.forEach(atom => {
        html += `
            <div class="atom-list-item" data-atom-index="${atom.index}">
                <span class="element">${atom.element}</span> (ID: ${atom.index})<br>
                <span style="font-size: 11px; color: #888;">
                    [${atom.position.x.toFixed(2)}, ${atom.position.y.toFixed(2)}, ${atom.position.z.toFixed(2)}]
                </span>
            </div>
        `;
    });
    
    atomListDiv.innerHTML = html;
    
    document.querySelectorAll('.atom-list-item').forEach(item => {
        item.addEventListener('click', () => {
            const index = parseInt(item.dataset.atomIndex);
            const atom = atoms.find(a => a.index === index);
            if (atom) {
                let targetMesh = null;
                meshToAtomMap.forEach((atomData, meshId) => {
                    if (atomData.index === index) {
                        targetMesh = findMeshById(meshId);
                    }
                });
                
                if (targetMesh) {
                    highlightAtom(targetMesh);
                    updateAtomInfoPanel(atom);
                    
                    controls.target.copy(atom.position);
                    camera.position.copy(atom.position.clone().add(new THREE.Vector3(2, 1, 2)));
                    controls.update();
                }
            }
        });
    });
}

function findMeshById(id) {
    let result = null;
    currentModelGroup.traverse((child) => {
        if (child.id === id) {
            result = child;
        }
    });
    return result;
}

function highlightAtom(mesh) {
    meshToAtomMap.forEach((data, id) => {
        const atomMesh = findMeshById(id);
        if (atomMesh && atomMesh.material) {
            if (Array.isArray(atomMesh.material)) {
                atomMesh.material.forEach(mat => {
                    mat.emissive = new THREE.Color(0x000000);
                });
            } else {
                atomMesh.material.emissive = new THREE.Color(0x000000);
            }
        }
    });
    
    if (mesh && mesh.material) {
        if (Array.isArray(mesh.material)) {
            mesh.material.forEach(mat => {
                mat.emissive = new THREE.Color(0x333333);
            });
        } else {
            mesh.material.emissive = new THREE.Color(0x333333);
        }
    }
}

function updateAtomInfoPanel(atom) {
    if (atom) {
        atomElementSpan.textContent = atom.element;
        atomIndexSpan.textContent = atom.index;
        atomPosXSpan.textContent = atom.position.x.toFixed(3);
        atomPosYSpan.textContent = atom.position.y.toFixed(3);
        atomPosZSpan.textContent = atom.position.z.toFixed(3);
        
        currentSelectedAtom = atom;
        atomModelBtn.disabled = false;
    } else {
        atomElementSpan.textContent = '—';
        atomIndexSpan.textContent = '—';
        atomPosXSpan.textContent = '—';
        atomPosYSpan.textContent = '—';
        atomPosZSpan.textContent = '—';
        
        atomModelBtn.disabled = true;
        currentSelectedAtom = null;
    }
}

function onClick(event) {
    if (atomModelGroup.visible) return;
    
    mouse.x = (event.clientX / renderer.domElement.clientWidth) * 2 - 1;
    mouse.y = -(event.clientY / renderer.domElement.clientHeight) * 2 + 1;
    
    raycaster.setFromCamera(mouse, camera);
    
    const meshes = [];
    currentModelGroup.traverse((child) => {
        if (child.isMesh) {
            meshes.push(child);
        }
    });
    
    const intersects = raycaster.intersectObjects(meshes);
    
    if (intersects.length > 0) {
        const hit = intersects[0];
        const mesh = hit.object;
        
        const atomData = meshToAtomMap.get(mesh.id);
        if (atomData) {
            updateAtomInfoPanel(atomData);
            highlightAtom(mesh);
            setStatus(`Выбран атом: ${atomData.element} (ID: ${atomData.index})`, false, false);
            console.log('Информация об атоме:', atomData);
        } else {
            setStatus('Выбран не атом', true, false);
        }
    }
}

async function loadModelFromZip(zipBlob) {
    clearModel();
    
    try {
        const JSZip = await import('https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js');
        
        let ZipConstructor;
        if (JSZip.default) {
            ZipConstructor = JSZip.default;
        } else if (typeof JSZip === 'function') {
            ZipConstructor = JSZip;
        } else {
            await loadScript('https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js');
            ZipConstructor = window.JSZip;
        }
        
        if (!ZipConstructor) {
            throw new Error('Не удалось загрузить JSZip');
        }
        
        const zip = new ZipConstructor();
        const loadedZip = await zip.loadAsync(zipBlob);
        
        const objFile = Object.values(loadedZip.files).find(file => file.name.endsWith('.obj'));
        const mtlFile = Object.values(loadedZip.files).find(file => file.name.endsWith('.mtl'));
        
        if (!objFile) {
            throw new Error('В ZIP архиве не найден .obj файл');
        }

        setStatus('Найден OBJ файл, загружаем...', false, true);
        
        const objText = await objFile.async('text');
        
        const atoms = parseAtomInfoFromObj(objText);
        console.log('Найдена информация об атомах:', atoms);
        
        if (mtlFile) {
            const mtlText = await mtlFile.async('text');
            
            const mtlBlob = new Blob([mtlText], { type: 'text/plain' });
            const objBlob = new Blob([objText], { type: 'text/plain' });
            
            const mtlUrl = URL.createObjectURL(mtlBlob);
            const objUrl = URL.createObjectURL(objBlob);
            
            const mtlLoader = new MTLLoader();
            
            mtlLoader.load(mtlUrl, (materials) => {
                materials.preload();
                
                const objLoader = new OBJLoader();
                objLoader.setMaterials(materials);
                
                objLoader.load(objUrl, (object) => {
                    const processedObject = processNestedObjects(object);
                    
                    processedObject.traverse((child) => {
                        if (child.isMesh) {
                            const groupName = child.name || (child.parent ? child.parent.name : '');
                            const atomMatch = groupName.match(/atom_(\w+)_(\d+)/);
                            
                            if (atomMatch) {
                                const element = atomMatch[1];
                                const index = parseInt(atomMatch[2]);
                                
                                const atom = atoms.find(a => a.index === index);
                                if (atom) {
                                    child.userData = {
                                        type: 'atom',
                                        element: element,
                                        index: index,
                                        position: atom.position.clone()
                                    };
                                    
                                    atomInfoMap.set(child.id, child.userData);
                                    meshToAtomMap.set(child.id, child.userData);
                                }
                            }
                        }
                    });
                    
                    const box = new THREE.Box3().setFromObject(processedObject);
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    if (maxDim > 3) {
                        processedObject.scale.setScalar(3 / maxDim);
                    } else if (maxDim < 0.5) {
                        processedObject.scale.setScalar(2);
                    }
                    
                    applySmoothingToGroup(processedObject);
                    
                    currentModelGroup.add(processedObject);
                    
                    savedMoleculeModel = processedObject.clone();
                    
                    savedAtomInfoMap = new Map();
                    savedMeshToAtomMap = new Map();
                    
                    atomInfoMap.forEach((value, key) => {
                        savedAtomInfoMap.set(key, value);
                    });
                    
                    meshToAtomMap.forEach((value, key) => {
                        savedMeshToAtomMap.set(key, value);
                    });
                    
                    centerCameraOnObject(processedObject);
                    
                    updateAtomList();
                    
                    setStatus(`Модель загружена! Найдено атомов: ${atomInfoMap.size}`, false, false);
                    
                    URL.revokeObjectURL(mtlUrl);
                    URL.revokeObjectURL(objUrl);
                }, undefined, (error) => {
                    setStatus('Ошибка загрузки OBJ', true, false);
                    console.error(error);
                    URL.revokeObjectURL(mtlUrl);
                    URL.revokeObjectURL(objUrl);
                });
            }, undefined, (error) => {
                setStatus('Ошибка загрузки MTL', true, false);
                console.error(error);
                URL.revokeObjectURL(mtlUrl);
                URL.revokeObjectURL(objUrl);
            });
        } else {
            const objBlob = new Blob([objText], { type: 'text/plain' });
            const objUrl = URL.createObjectURL(objBlob);
            
            const objLoader = new OBJLoader();
            objLoader.load(objUrl, (object) => {
                const processedObject = processNestedObjects(object);
                
                processedObject.traverse((child) => {
                    if (child.isMesh) {
                        const groupName = child.name || (child.parent ? child.parent.name : '');
                        const atomMatch = groupName.match(/atom_(\w+)_(\d+)/);
                        
                        if (atomMatch) {
                            const element = atomMatch[1];
                            const index = parseInt(atomMatch[2]);
                            
                            const atom = atoms.find(a => a.index === index);
                            if (atom) {
                                child.userData = {
                                    type: 'atom',
                                    element: element,
                                    index: index,
                                    position: atom.position.clone()
                                };
                                
                                atomInfoMap.set(child.id, child.userData);
                                meshToAtomMap.set(child.id, child.userData);
                            }
                        }
                    }
                });
                
                const box = new THREE.Box3().setFromObject(processedObject);
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                if (maxDim > 3) {
                    processedObject.scale.setScalar(3 / maxDim);
                } else if (maxDim < 0.5) {
                    processedObject.scale.setScalar(2);
                }
                
                applySmoothingToGroup(processedObject);
                
                currentModelGroup.add(processedObject);
                
                savedMoleculeModel = processedObject.clone();
                
                savedAtomInfoMap = new Map();
                savedMeshToAtomMap = new Map();
                
                atomInfoMap.forEach((value, key) => {
                    savedAtomInfoMap.set(key, value);
                });
                
                meshToAtomMap.forEach((value, key) => {
                    savedMeshToAtomMap.set(key, value);
                });
                
                centerCameraOnObject(processedObject);
                
                updateAtomList();
                
                setStatus(`Модель загружена! Найдено атомов: ${atomInfoMap.size}`, false, false);
                URL.revokeObjectURL(objUrl);
            }, undefined, (error) => {
                setStatus('Ошибка загрузки OBJ', true, false);
                console.error(error);
                URL.revokeObjectURL(objUrl);
            });
        }
    } catch (error) {
        console.error('Ошибка при работе с ZIP:', error);
        setStatus(`Ошибка распаковки ZIP: ${error.message}`, true, false);
    }
}

function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

function centerCameraOnObject(object) {
    const box = new THREE.Box3().setFromObject(object);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    
    controls.target.copy(center);
    
    const distance = Math.max(size * 1.5, 3);
    const direction = new THREE.Vector3(1, 0.5, 1).normalize();
    camera.position.copy(center.clone().add(direction.multiplyScalar(distance)));
    controls.update();
}

async function sendFormula() {
    const formula = inputElement.value.trim();
    if (!formula) {
        setStatus('Введите формулу', true, false);
        return;
    }

    sendButton.disabled = true;
    setStatus('Отправка запроса...', false, true);

    try {
        const baseUrl = CONFIG.MODEL_GENERATOR_API;
        const url = new URL(baseUrl);
        url.searchParams.append('chemistry_formule', formula);
        
        console.log('Fetching:', url.toString());
        
        const response = await fetch(url.toString(), {
            method: 'GET',
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const blob = await response.blob();
        
        if (blob.size === 0) {
            throw new Error('Получен пустой файл');
        }

        setStatus('Архив получен, загружаем модель...', false, true);
        
        await loadModelFromZip(blob);
        
        currentModelGroup.visible = true;
        atomModelGroup.visible = false;
        backToMoleculeBtn.style.display = 'none';
        viewToggle.style.display = 'none';
        
    } catch (error) {
        console.error('Error:', error);
        setStatus(`Ошибка: ${error.message}`, true, false);
    } finally {
        sendButton.disabled = false;
    }
}

sendButton.addEventListener('click', sendFormula);

inputElement.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendFormula();
    }
});

renderer.domElement.addEventListener('click', onClick);

atomListBtn.addEventListener('click', () => {
    atomListDiv.classList.toggle('visible');
    if (atomListDiv.classList.contains('visible')) {
        updateAtomList();
    }
});

atomModelBtn.addEventListener('click', async () => {
    if (currentSelectedAtom) {
        await loadAtomModelInScene(currentSelectedAtom.element);
    }
});

function animate() {
    requestAnimationFrame(animate);
    
    controls.update();
    
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', onWindowResize, false);
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function createDemoModel() {
    const demoGroup = new THREE.Group();
    
    const centerGeo = new THREE.SphereGeometry(0.8, 64, 32);
    const centerMat = new THREE.MeshStandardMaterial({ 
        color: 0x6a5acd, 
        emissive: 0x221133,
        roughness: 0.2,
        metalness: 0.1,
        emissiveIntensity: 0.3
    });
    const centerSphere = new THREE.Mesh(centerGeo, centerMat);
    centerSphere.position.set(0, 0.5, 0);
    centerSphere.userData = { type: 'atom', element: 'C', index: 0, position: new THREE.Vector3(0, 0.5, 0) };
    demoGroup.add(centerSphere);
    
    atomInfoMap.set(centerSphere.id, centerSphere.userData);
    meshToAtomMap.set(centerSphere.id, centerSphere.userData);
    
    const positions = [
        { pos: [1.5, 0.8, 0.8], element: 'O', color: 0xff5555, index: 1 },
        { pos: [-1.2, 0.3, -1.2], element: 'N', color: 0x55ff55, index: 2 },
        { pos: [0.7, -0.2, 1.8], element: 'H', color: 0xffffff, index: 3 },
        { pos: [-1.5, 1.0, 1.0], element: 'C', color: 0xffff55, index: 4 },
        { pos: [1.2, -0.1, -1.5], element: 'O', color: 0xff55ff, index: 5 }
    ];
    
    positions.forEach((p) => {
        const sphereGeo = new THREE.SphereGeometry(0.35, 64, 32);
        const sphereMat = new THREE.MeshStandardMaterial({ 
            color: p.color,
            emissive: new THREE.Color(p.color).multiplyScalar(0.2),
            roughness: 0.2,
            metalness: 0.1
        });
        const sphere = new THREE.Mesh(sphereGeo, sphereMat);
        sphere.position.set(p.pos[0], p.pos[1], p.pos[2]);
        sphere.userData = { type: 'atom', element: p.element, index: p.index, position: new THREE.Vector3(p.pos[0], p.pos[1], p.pos[2]) };
        demoGroup.add(sphere);
        
        atomInfoMap.set(sphere.id, sphere.userData);
        meshToAtomMap.set(sphere.id, sphere.userData);
    });
    
    currentModelGroup.add(demoGroup);
    
    savedMoleculeModel = demoGroup.clone();
    savedAtomInfoMap = new Map();
    savedMeshToAtomMap = new Map();
    
    atomInfoMap.forEach((value, key) => {
        savedAtomInfoMap.set(key, value);
    });
    
    meshToAtomMap.forEach((value, key) => {
        savedMeshToAtomMap.set(key, value);
    });
    
    updateAtomList();
}

createDemoModel();
setStatus('Демо-модель с атомами. Кликните на атом для информации.', false, false);