#!/usr/bin/env python3
"""
model_viewer.py

A clean and simple Streamlit app for viewing 3D models from any folder.

Usage:
    python model_viewer.py
    # or with custom port:
    MODEL_VIEWER_PORT=8503 python model_viewer.py
"""

import streamlit as st
import streamlit.components.v1 as components
import trimesh
import os
import numpy as np
import glob
import tempfile
import json
import base64
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="Model Viewer - 3D Model Viewer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_glb_viewer(file_path, height=600):
    """Render a GLB/GLTF file using Three.js viewer embedded in Streamlit."""
    
    try:
        # Load file with trimesh to get mesh data
        scene_or_mesh = trimesh.load(file_path)
        
        # Handle both Scene and Mesh objects
        if hasattr(scene_or_mesh, 'geometry'):  # Scene object
            # For Scene objects, combine all meshes into one
            vertices_list = []
            faces_list = []
            colors_list = []
            face_offset = 0
            
            for geometry_name, geometry in scene_or_mesh.geometry.items():
                if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                    vertices_list.append(geometry.vertices)
                    faces_list.append(geometry.faces + face_offset)
                    
                    # Get colors if available
                    if hasattr(geometry.visual, 'vertex_colors') and geometry.visual.vertex_colors is not None:
                        colors_list.append(geometry.visual.vertex_colors)
                    elif hasattr(geometry.visual, 'face_colors') and geometry.visual.face_colors is not None:
                        colors_list.append(geometry.visual.face_colors)
                    
                    face_offset += len(geometry.vertices)
            
            if vertices_list:
                vertices = np.vstack(vertices_list)
                faces = np.vstack(faces_list)
                colors = np.vstack(colors_list) if colors_list else None
            else:
                raise ValueError("No valid mesh data found in GLB file")
        else:  # Single Mesh object
            vertices = scene_or_mesh.vertices
            faces = scene_or_mesh.faces
            
            # Get colors if available (prefer vertex colors over face colors)
            colors = None
            if hasattr(scene_or_mesh.visual, 'vertex_colors') and scene_or_mesh.visual.vertex_colors is not None:
                colors = scene_or_mesh.visual.vertex_colors
            elif hasattr(scene_or_mesh.visual, 'face_colors') and scene_or_mesh.visual.face_colors is not None:
                colors = scene_or_mesh.visual.face_colors
        
        # Create Three.js viewer HTML
        viewer_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; padding: 0; overflow: hidden; }}
                #canvas {{ width: 100%; height: {height}px; }}
                #controls {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background: rgba(0,0,0,0.7);
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    z-index: 1000;
                }}
            </style>
        </head>
        <body>
            <div id="controls">
                🖱️ Left-click: Rotate | Right-click: Pan | Scroll: Zoom
            </div>
            <div id="canvas"></div>
            
            <script type="importmap">
            {{
                "imports": {{
                    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
                }}
            }}
            </script>
            
            <script type="module">
                import * as THREE from 'three';
                import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
                
                // Scene setup
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x000000);
                
                const camera = new THREE.PerspectiveCamera(75, window.innerWidth / {height}, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, {height});
                renderer.outputColorSpace = THREE.SRGBColorSpace;
                renderer.toneMapping = THREE.ACESFilmicToneMapping;
                renderer.toneMappingExposure = 1.0;
                document.getElementById('canvas').appendChild(renderer.domElement);
                
                // Simple ambient light for basic visibility
                const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
                scene.add(ambientLight);
                
                // Controls
                const controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.enableZoom = true;
                controls.enablePan = true;
                controls.enableRotate = true;
                
                // Mesh data
                const vertices = {vertices.tolist()};
                const faces = {faces.tolist()};
                const colors = {colors.tolist() if colors is not None else 'null'};
                
                // Debug color information
                console.log('Colors available:', colors !== null);
                if (colors && colors.length > 0) {{
                    console.log('Color sample:', colors.slice(0, 3));
                    console.log('Color dimensions:', colors[0].length);
                }}
                
                // Create geometry
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices.flat(), 3));
                
                // Handle faces (convert to triangles if needed)
                let triangles = [];
                for (let face of faces) {{
                    if (face.length === 3) {{
                        triangles.push(face[0], face[1], face[2]);
                    }} else if (face.length === 4) {{
                        // Convert quad to two triangles
                        triangles.push(face[0], face[1], face[2]);
                        triangles.push(face[0], face[2], face[3]);
                    }}
                }}
                geometry.setIndex(triangles);
                
                // Add colors if available
                if (colors && colors.length > 0) {{
                    if (colors[0].length >= 3) {{
                        // Vertex colors (handle both RGB and RGBA)
                        const colorData = colors.map(c => c.slice(0, 3)); // Take only RGB, ignore alpha
                        const colorArray = new Float32Array(colorData.flat().map(c => c / 255.0)); // Normalize to 0-1
                        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colorArray, 3));
                    }}
                }}
                
                // Create material with flat voxel appearance
                const material = new THREE.MeshBasicMaterial({{ 
                    vertexColors: colors && colors.length > 0,
                    side: THREE.DoubleSide,
                    color: colors && colors.length > 0 ? 0xffffff : 0x888888,
                    transparent: false
                }});
                
                // Create mesh
                const mesh = new THREE.Mesh(geometry, material);
                scene.add(mesh);
                
                // Center and scale the model
                const box = new THREE.Box3().setFromObject(mesh);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 5 / maxDim;
                
                mesh.scale.set(scale, scale, scale);
                mesh.position.sub(center.multiplyScalar(scale));
                
                // Position camera
                camera.position.set(0, 2, 8);
                camera.lookAt(0, 0, 0);
                controls.update();
                
                // Animation loop
                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();
                
                // Handle resize
                window.addEventListener('resize', function() {{
                    camera.aspect = window.innerWidth / {height};
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, {height});
                }});
            </script>
        </body>
        </html>
        """
        
        components.html(viewer_html, height=height + 50)
        
    except Exception as e:
        st.error(f"Error rendering model file: {e}")

def load_model_info(file_path):
    """Load model file and return mesh information."""
    try:
        scene_or_mesh = trimesh.load(file_path)
        
        # Handle both Scene and Mesh objects
        if hasattr(scene_or_mesh, 'geometry'):  # Scene object
            # For Scene objects, combine all meshes into one
            vertices_list = []
            faces_list = []
            colors_list = []
            face_offset = 0
            total_volume = 0
            total_surface_area = 0
            is_watertight = True
            
            for geometry_name, geometry in scene_or_mesh.geometry.items():
                if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                    vertices_list.append(geometry.vertices)
                    faces_list.append(geometry.faces + face_offset)
                    
                    # Get colors if available
                    if hasattr(geometry.visual, 'vertex_colors') and geometry.visual.vertex_colors is not None:
                        colors_list.append(geometry.visual.vertex_colors)
                    elif hasattr(geometry.visual, 'face_colors') and geometry.visual.face_colors is not None:
                        colors_list.append(geometry.visual.face_colors)
                    
                    # Aggregate properties
                    if hasattr(geometry, 'volume'):
                        total_volume += geometry.volume if geometry.is_watertight else 0
                    if hasattr(geometry, 'area'):
                        total_surface_area += geometry.area
                    if not geometry.is_watertight:
                        is_watertight = False
                    
                    face_offset += len(geometry.vertices)
            
            if vertices_list:
                vertices = np.vstack(vertices_list)
                faces = np.vstack(faces_list)
                colors = np.vstack(colors_list) if colors_list else None
                
                info = {
                    'mesh': scene_or_mesh,
                    'vertices': vertices,
                    'faces': faces,
                    'vertex_count': len(vertices),
                    'face_count': len(faces),
                    'bounds': scene_or_mesh.bounds,
                    'volume': total_volume if is_watertight else None,
                    'surface_area': total_surface_area,
                    'is_watertight': is_watertight
                }
            else:
                raise ValueError("No valid mesh data found in model file")
        else:  # Single Mesh object
            vertices = scene_or_mesh.vertices
            faces = scene_or_mesh.faces
            
            info = {
                'mesh': scene_or_mesh,
                'vertices': vertices,
                'faces': faces,
                'vertex_count': len(vertices),
                'face_count': len(faces),
                'bounds': scene_or_mesh.bounds,
                'volume': scene_or_mesh.volume if scene_or_mesh.is_watertight else None,
                'surface_area': scene_or_mesh.area,
                'is_watertight': scene_or_mesh.is_watertight
            }
            
            colors = None
            if hasattr(scene_or_mesh.visual, 'vertex_colors') and scene_or_mesh.visual.vertex_colors is not None:
                colors = scene_or_mesh.visual.vertex_colors
            elif hasattr(scene_or_mesh.visual, 'face_colors') and scene_or_mesh.visual.face_colors is not None:
                colors = scene_or_mesh.visual.face_colors
        
        # Check for colors
        if colors is not None:
            info['has_vertex_colors'] = True
            info['vertex_colors'] = colors
        else:
            info['has_vertex_colors'] = False
            
        info['has_face_colors'] = False  # We're not handling face colors separately in this implementation
        
        return info
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None

def get_available_model_folders():
    """Get list of available model folders from ./output/models."""
    models_dir = "./output/models"
    
    if not os.path.exists(models_dir):
        return []
    
    # Get all subdirectories in ./output/models
    available_folders = []
    for item in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, item)
        if os.path.isdir(folder_path):
            available_folders.append(folder_path)
    
    return sorted(available_folders)

def load_gltf_with_textures(gltf_path):
    """Load GLTF file and return model data with proper texture handling."""
    try:
        # Load the GLTF file
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        model_folder = os.path.dirname(gltf_path)
        
        # Load mesh data using trimesh (more reliable for geometry)
        scene_or_mesh = trimesh.load(gltf_path)
        
        # Extract mesh data
        vertices = []
        faces = []
        uvs = []
        colors = []
        
        if hasattr(scene_or_mesh, 'geometry'):  # Scene object
            vertices_list = []
            faces_list = []
            uvs_list = []
            colors_list = []
            face_offset = 0
            
            for geometry_name, geometry in scene_or_mesh.geometry.items():
                if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                    vertices_list.append(geometry.vertices)
                    faces_list.append(geometry.faces + face_offset)
                    
                    # Get UV coordinates if available
                    if hasattr(geometry.visual, 'uv') and geometry.visual.uv is not None:
                        uvs_list.append(geometry.visual.uv)
                    
                    # Get colors if available
                    if hasattr(geometry.visual, 'vertex_colors') and geometry.visual.vertex_colors is not None:
                        colors_list.append(geometry.visual.vertex_colors)
                    elif hasattr(geometry.visual, 'face_colors') and geometry.visual.face_colors is not None:
                        colors_list.append(geometry.visual.face_colors)
                    
                    face_offset += len(geometry.vertices)
            
            if vertices_list:
                vertices = np.vstack(vertices_list)
                faces = np.vstack(faces_list)
                if uvs_list:
                    uvs = np.vstack(uvs_list)
                if colors_list:
                    colors = np.vstack(colors_list)
        else:  # Single Mesh object
            vertices = scene_or_mesh.vertices
            faces = scene_or_mesh.faces
            
            # Get UV coordinates if available
            if hasattr(scene_or_mesh.visual, 'uv') and scene_or_mesh.visual.uv is not None:
                uvs = scene_or_mesh.visual.uv
            
            # Get colors if available
            if hasattr(scene_or_mesh.visual, 'vertex_colors') and scene_or_mesh.visual.vertex_colors is not None:
                colors = scene_or_mesh.visual.vertex_colors
            elif hasattr(scene_or_mesh.visual, 'face_colors') and scene_or_mesh.visual.face_colors is not None:
                colors = scene_or_mesh.visual.face_colors
        
        # Extract texture information
        textures_info = {}
        if 'images' in gltf_data:
            for i, image in enumerate(gltf_data['images']):
                if 'uri' in image:
                    texture_path = os.path.join(model_folder, image['uri'])
                    if os.path.exists(texture_path):
                        # Load texture as base64 for embedding
                        with open(texture_path, 'rb') as img_file:
                            img_data = img_file.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            # Determine MIME type from file extension
                            ext = os.path.splitext(image['uri'])[1].lower()
                            mime_type = f"image/{ext[1:]}" if ext in ['.png', '.jpg', '.jpeg'] else "image/png"
                            textures_info[i] = f"data:{mime_type};base64,{img_base64}"
        
        # Extract material information
        materials_info = {}
        if 'materials' in gltf_data:
            for i, material in enumerate(gltf_data['materials']):
                materials_info[i] = {
                    'name': material.get('name', f'material_{i}'),
                    'doubleSided': material.get('doubleSided', False),
                }
                
                # Handle PBR material properties
                if 'pbrMetallicRoughness' in material:
                    pbr = material['pbrMetallicRoughness']
                    materials_info[i]['baseColorFactor'] = pbr.get('baseColorFactor', [1, 1, 1, 1])
                    if 'baseColorTexture' in pbr:
                        tex_index = pbr['baseColorTexture']['index']
                        materials_info[i]['baseColorTexture'] = textures_info.get(tex_index)
                
                # Handle extensions (like specular glossiness)
                if 'extensions' in material:
                    if 'KHR_materials_pbrSpecularGlossiness' in material['extensions']:
                        sg = material['extensions']['KHR_materials_pbrSpecularGlossiness']
                        materials_info[i]['diffuseFactor'] = sg.get('diffuseFactor', [1, 1, 1, 1])
                        if 'diffuseTexture' in sg:
                            tex_index = sg['diffuseTexture']['index']
                            materials_info[i]['diffuseTexture'] = textures_info.get(tex_index)
        
        return {
            'gltf_data': gltf_data,
            'textures': textures_info,
            'materials': materials_info,
            'model_folder': model_folder,
            'vertices': vertices,
            'faces': faces,
            'uvs': uvs,
            'colors': colors
        }
    except Exception as e:
        st.error(f"Error loading GLTF with textures: {e}")
        return None

def create_threejs_viewer_with_textures(gltf_info, height=600):
    """Create Three.js viewer HTML using proper GLTFLoader."""
    
    model_folder = gltf_info['model_folder']
    
    # Get the GLTF file path relative to the model folder
    gltf_file_name = "scene.gltf"  # Default name
    if os.path.exists(os.path.join(model_folder, "scene.gltf")):
        gltf_file_name = "scene.gltf"
    elif os.path.exists(os.path.join(model_folder, "scene.glb")):
        gltf_file_name = "scene.glb"
    
    # Create a data URL for the GLTF file
    gltf_path = os.path.join(model_folder, gltf_file_name)
    with open(gltf_path, 'rb') as f:
        gltf_data_bytes = f.read()
    
    gltf_base64 = base64.b64encode(gltf_data_bytes).decode('utf-8')
    gltf_data_url = f"data:model/gltf-binary;base64,{gltf_base64}"
    
    viewer_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; }}
            #canvas {{ width: 100%; height: {height}px; }}
            #controls {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-family: Arial, sans-serif;
                font-size: 12px;
                z-index: 1000;
            }}
            #loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-size: 18px;
                z-index: 1001;
            }}
        </style>
    </head>
    <body>
        <div id="controls">
            🖱️ Left-click: Rotate | Right-click: Pan | Scroll: Zoom
        </div>
        <div id="loading">Loading 3D model...</div>
        <div id="canvas"></div>
        
        <script type="importmap">
        {{
            "imports": {{
                "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
                "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
            }}
        }}
        </script>
        
        <script type="module">
            import * as THREE from 'three';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
            import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
            
            // Scene setup
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x000000);
            
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / {height}, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, {height});
            renderer.outputColorSpace = THREE.SRGBColorSpace;
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1.0;
            document.getElementById('canvas').appendChild(renderer.domElement);
            
            // Lighting setup for PBR materials
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 5, 5);
            directionalLight.castShadow = true;
            scene.add(directionalLight);
            
            const hemisphereLight = new THREE.HemisphereLight(0x87CEEB, 0x362d1d, 0.3);
            scene.add(hemisphereLight);
            
            // Controls
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.enableZoom = true;
            controls.enablePan = true;
            controls.enableRotate = true;
            
            {texture_loading_code}
            {material_creation_code}
            
            // Debug materials and textures
            console.log('Available textures:', Object.keys(textures));
            console.log('Available materials:', Object.keys(materials));
            if (Object.keys(materials).length > 0) {{
                console.log('First material:', materials[0]);
            }}
            
            // Load model using trimesh data (more reliable)
            const model = new THREE.Group();
            
            // Get mesh data from trimesh (passed from Python)
            const meshData = {json.dumps({
                'vertices': gltf_info['vertices'].tolist() if len(gltf_info['vertices']) > 0 else [],
                'faces': gltf_info['faces'].tolist() if len(gltf_info['faces']) > 0 else [],
                'uvs': gltf_info['uvs'].tolist() if len(gltf_info['uvs']) > 0 else [],
                'colors': gltf_info['colors'].tolist() if len(gltf_info['colors']) > 0 else []
            })};
            
            // Debug mesh data
            console.log('Mesh data loaded:', {{
                vertices: meshData.vertices.length,
                faces: meshData.faces.length,
                uvs: meshData.uvs.length,
                colors: meshData.colors.length
            }});
            
            if (meshData.vertices && meshData.vertices.length > 0) {{
                console.log('Creating geometry from mesh data...');
                console.log('Vertices count:', meshData.vertices.length);
                console.log('Faces count:', meshData.faces ? meshData.faces.length : 0);
                
                // Create geometry from mesh data
                const geometry = new THREE.BufferGeometry();
                
                // Set vertices
                const vertices = new Float32Array(meshData.vertices.flat());
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                console.log('Vertices set:', vertices.length / 3, 'vertices');
                
                // Handle faces (convert to triangles if needed)
                let triangles = [];
                if (meshData.faces && meshData.faces.length > 0) {{
                    console.log('Processing', meshData.faces.length, 'faces...');
                    for (let face of meshData.faces) {{
                        if (face.length === 3) {{
                            triangles.push(face[0], face[1], face[2]);
                        }} else if (face.length === 4) {{
                            // Convert quad to two triangles
                            triangles.push(face[0], face[1], face[2]);
                            triangles.push(face[0], face[2], face[3]);
                        }}
                    }}
                    geometry.setIndex(triangles);
                    console.log('Faces processed:', triangles.length / 3, 'triangles');
                }} else {{
                    console.log('No faces data available, creating non-indexed geometry');
                }}
                
                // Set UV coordinates if available
                if (meshData.uvs && meshData.uvs.length > 0) {{
                    const uvs = new Float32Array(meshData.uvs.flat());
                    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
                    console.log('UV coordinates set:', uvs.length / 2, 'UV pairs');
                }} else {{
                    console.log('No UV coordinates available');
                }}
                
                // Compute normals if not present
                if (!geometry.attributes.normal) {{
                    geometry.computeVertexNormals();
                    console.log('Computed vertex normals');
                }}
                
                // Create material (use first available material or default)
                const materialIndex = 0;
                const material = materials[materialIndex] || new THREE.MeshStandardMaterial({{
                    color: 0x888888,
                    side: THREE.DoubleSide
                }});
                
                // Debug material
                console.log('Using material:', material);
                console.log('Material properties:', {{
                    name: material.name,
                    map: material.map ? 'texture loaded' : 'no texture',
                    color: material.color,
                    side: material.side
                }});
                
                // Create mesh
                const mesh = new THREE.Mesh(geometry, material);
                console.log('Mesh created with geometry:', geometry);
                console.log('Mesh bounding box:', geometry.boundingBox);
                console.log('Mesh vertices count:', geometry.attributes.position.count);
                
                model.add(mesh);
                console.log('Mesh added to model group');
            }} else {{
                // Fallback: create a simple box
                const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
                const material = new THREE.MeshStandardMaterial({{ color: 0x888888 }});
                const mesh = new THREE.Mesh(boxGeometry, material);
                model.add(mesh);
            }}
            
            scene.add(model);
            
            // Center and scale the model
            const box = new THREE.Box3().setFromObject(model);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 5 / maxDim;
            
            console.log('Model bounds:', {{
                center: center,
                size: size,
                maxDimension: maxDim,
                scale: scale
            }});
            
            model.scale.set(scale, scale, scale);
            model.position.sub(center.multiplyScalar(scale));
            
            console.log('Model positioned at:', model.position);
            console.log('Model scale:', model.scale);
            
            // Position camera
            camera.position.set(0, 2, 8);
            camera.lookAt(0, 0, 0);
            controls.update();
            
            console.log('Camera positioned at:', camera.position);
            console.log('Camera looking at:', new THREE.Vector3(0, 0, 0));
            
            // Animation loop
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
            
            // Handle resize
            window.addEventListener('resize', function() {{
                camera.aspect = window.innerWidth / {height};
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, {height});
            }});
        </script>
    </body>
    </html>
    """
    
    return viewer_html

# --- Main App ---

# Sidebar
with st.sidebar:
    st.title("🎨 Model Viewer")
    st.markdown("A clean and simple 3D viewer for model files")
    
    st.divider()
    st.header("📁 Folder Selection")
    
    # Get available model folders
    available_folders = get_available_model_folders()
    
    if available_folders:
        # Create display names for folders (just the folder name)
        folder_display_names = [os.path.basename(folder) for folder in available_folders]
        folder_options = ["Select a model..."] + folder_display_names
        
        selected_folder_name = st.selectbox(
            "Choose model:",
            folder_options,
            help="Select a model from the available model folders"
        )
        
        if selected_folder_name != "Select a model...":
            # Find the full path for the selected folder name
            selected_folder = None
            for folder in available_folders:
                if os.path.basename(folder) == selected_folder_name:
                    selected_folder = folder
                    break
        else:
            selected_folder = None
    else:
        st.warning("No model folders found in ./output/models/")
        selected_folder = None
    
    # Model loading from selected folder
    selected_file = None
    temp_file = None
    
    if selected_folder:
        st.divider()
        st.header("📄 Model Info")
        
        # Look for the main model file in the folder
        # Priority: scene.gltf > scene.glb > first .gltf > first .glb > first .obj > etc.
        # (GLTF preferred for texture support)
        model_file = None
        
        # Check for scene.gltf first (preferred for texture support)
        scene_gltf_path = os.path.join(selected_folder, "scene.gltf")
        if os.path.exists(scene_gltf_path):
            model_file = scene_gltf_path
        
        # If no scene.gltf, check for scene.glb
        if not model_file:
            scene_glb_path = os.path.join(selected_folder, "scene.glb")
            if os.path.exists(scene_glb_path):
                model_file = scene_glb_path
        
        # If still no main file, look for any model files (prioritize GLTF for texture support)
        if not model_file:
            supported_extensions = ['gltf', 'glb', 'obj', 'ply', 'stl', 'dae']
            for ext in supported_extensions:
                pattern = os.path.join(selected_folder, f"*.{ext}")
                files = glob.glob(pattern)
                if files:
                    model_file = files[0]  # Take the first one found
                    break
        
        if model_file:
            selected_file = model_file
            model_name = os.path.basename(selected_folder)
            st.success(f"Selected model: **{model_name}**")
            st.caption(f"Loading: {os.path.basename(model_file)}")
            
            # Show available files in the folder
            all_files = os.listdir(selected_folder)
            if len(all_files) > 1:
                st.caption(f"Folder contains {len(all_files)} files")
            
            # Debug info for GLTF files
            if model_file.endswith('.gltf'):
                try:
                    with open(model_file, 'r') as f:
                        gltf_data = json.load(f)
                    
                    # Show texture info
                    if 'images' in gltf_data:
                        st.caption(f"Found {len(gltf_data['images'])} texture(s)")
                        for i, img in enumerate(gltf_data['images']):
                            if 'uri' in img:
                                st.caption(f"  - Texture {i}: {img['uri']}")
                    
                    # Show material info
                    if 'materials' in gltf_data:
                        st.caption(f"Found {len(gltf_data['materials'])} material(s)")
                        for i, mat in enumerate(gltf_data['materials']):
                            mat_name = mat.get('name', f'material_{i}')
                            st.caption(f"  - Material {i}: {mat_name}")
                            
                            # Check for texture references
                            if 'pbrMetallicRoughness' in mat and 'baseColorTexture' in mat['pbrMetallicRoughness']:
                                tex_idx = mat['pbrMetallicRoughness']['baseColorTexture']['index']
                                st.caption(f"    → Uses baseColorTexture {tex_idx}")
                            
                            if 'extensions' in mat and 'KHR_materials_pbrSpecularGlossiness' in mat['extensions']:
                                sg = mat['extensions']['KHR_materials_pbrSpecularGlossiness']
                                if 'diffuseTexture' in sg:
                                    tex_idx = sg['diffuseTexture']['index']
                                    st.caption(f"    → Uses diffuseTexture {tex_idx}")
                except Exception as e:
                    st.caption(f"Debug info error: {e}")
        else:
            st.error(f"No model files found in {os.path.basename(selected_folder)}")
            st.caption("Expected formats: scene.glb, scene.gltf, or other 3D model files")

# Main content
if selected_file and os.path.exists(selected_file):
    # Check if it's a GLTF file that should use texture loading
    file_ext = os.path.splitext(selected_file)[1].lower()
    
    if file_ext == '.gltf':
        # Load GLTF with proper texture support
        gltf_info = load_gltf_with_textures(selected_file)
        if gltf_info:
            # Create viewer with texture support
            viewer_html = create_threejs_viewer_with_textures(gltf_info, height=600)
            components.html(viewer_html, height=650)
    else:
        # Load mesh information (needed for rendering)
        mesh_info = load_model_info(selected_file)
        
        if mesh_info:
            # 3D Viewer
            render_glb_viewer(selected_file, height=600)
    
    # Clean up temporary file
    if temp_file and os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass
            
else:
    # Display example information in sidebar
    with st.sidebar:
        st.divider()
        if selected_folder:
            st.info("👆 Model folder selected, but no valid model file found.")
        else:
            st.info("👆 Select a model from the dropdown to get started.")

# Port configuration (if running as main)
if __name__ == "__main__":
    import sys
    port = int(os.environ.get('MODEL_VIEWER_PORT', 8503))
    sys.argv.extend(['--server.port', str(port)])
