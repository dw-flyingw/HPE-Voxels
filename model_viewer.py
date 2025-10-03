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
                renderer.toneMappingExposure = 1.5;
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

def extract_uvs_from_gltf(gltf_data, model_folder):
    """Extract UV coordinates from GLTF accessors."""
    try:
        # Look for TEXCOORD_0 accessor in meshes
        uvs = []
        
        if 'meshes' in gltf_data:
            for mesh in gltf_data['meshes']:
                for primitive in mesh.get('primitives', []):
                    attributes = primitive.get('attributes', {})
                    
                    # Look for TEXCOORD_0 attribute
                    if 'TEXCOORD_0' in attributes:
                        texcoord_index = attributes['TEXCOORD_0']
                        accessor = gltf_data['accessors'][texcoord_index]
                        
                        # Load UV data from buffer
                        buffer_view_index = accessor['bufferView']
                        buffer_view = gltf_data['bufferViews'][buffer_view_index]
                        buffer_index = buffer_view['buffer']
                        buffer_info = gltf_data['buffers'][buffer_index]
                        
                        # Read buffer data
                        buffer_path = os.path.join(model_folder, buffer_info['uri'])
                        with open(buffer_path, 'rb') as f:
                            buffer_data = f.read()
                        
                        # Extract UV coordinates
                        byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
                        uv_data = buffer_data[byte_offset:byte_offset + buffer_view['byteLength']]
                        
                        # Convert to numpy array (assuming FLOAT32)
                        uv_array = np.frombuffer(uv_data, dtype=np.float32)
                        uv_array = uv_array.reshape(-1, 2)  # Each UV pair has 2 components
                        
                        uvs.extend(uv_array)
        
        print(f"UV extraction result: {len(uvs)} UV coordinates found")
        return np.array(uvs) if uvs else np.array([])
    except Exception as e:
        print(f"Error extracting UVs from GLTF: {e}")
        return np.array([])

def find_texture_in_folder(model_folder):
    """Find texture files in model folder as fallback."""
    texture_extensions = ['.png', '.jpg', '.jpeg', '.tga', '.bmp']
    texture_files = []
    
    # Check textures subfolder
    textures_folder = os.path.join(model_folder, 'textures')
    if os.path.exists(textures_folder):
        for file in os.listdir(textures_folder):
            if any(file.lower().endswith(ext) for ext in texture_extensions):
                texture_files.append(os.path.join('textures', file))
    
    # Check root folder
    for file in os.listdir(model_folder):
        if any(file.lower().endswith(ext) for ext in texture_extensions):
            texture_files.append(file)
    
    return texture_files

def load_gltf_with_textures(gltf_path):
    """Load GLTF file and return model data with proper texture handling."""
    try:
        # Load the GLTF file
        with open(gltf_path, 'r') as f:
            gltf_data = json.load(f)
        
        model_folder = os.path.dirname(gltf_path)
        
        # Load mesh data using trimesh (more reliable for geometry)
        print(f"Loading GLTF file: {gltf_path}")
        scene_or_mesh = trimesh.load(gltf_path)
        print(f"Loaded mesh: {type(scene_or_mesh)}")
        
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
                print(f"Scene: {len(vertices)} vertices, {len(faces)} faces")
            else:
                print("No vertices found in scene")
        else:  # Single Mesh object
            vertices = scene_or_mesh.vertices
            faces = scene_or_mesh.faces
            print(f"Single mesh: {len(vertices)} vertices, {len(faces)} faces")
            
            # Get UV coordinates if available
            if hasattr(scene_or_mesh.visual, 'uv') and scene_or_mesh.visual.uv is not None:
                uvs = scene_or_mesh.visual.uv
            
            # Get colors if available
            if hasattr(scene_or_mesh.visual, 'vertex_colors') and scene_or_mesh.visual.vertex_colors is not None:
                colors = scene_or_mesh.visual.vertex_colors
            elif hasattr(scene_or_mesh.visual, 'face_colors') and scene_or_mesh.visual.face_colors is not None:
                colors = scene_or_mesh.visual.face_colors
        
        # If no UVs extracted from trimesh, try to extract from GLTF directly
        if len(uvs) == 0:
            print("No UVs from trimesh, attempting GLTF extraction...")
            uvs = extract_uvs_from_gltf(gltf_data, model_folder)
            print(f"GLTF UV extraction result: {len(uvs)} UVs")
        
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
        
        # If no textures found in GLTF, search for textures in folder as fallback
        if not textures_info:
            texture_files = find_texture_in_folder(model_folder)
            for i, texture_file in enumerate(texture_files):
                texture_path = os.path.join(model_folder, texture_file)
                if os.path.exists(texture_path):
                    with open(texture_path, 'rb') as img_file:
                        img_data = img_file.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        ext = os.path.splitext(texture_file)[1].lower()
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
        
        # If no materials found or no textures in materials, create default material with fallback texture
        if not materials_info or not any(mat.get('baseColorTexture') or mat.get('diffuseTexture') for mat in materials_info.values()):
            if textures_info:
                # Use first available texture for default material
                default_texture = list(textures_info.values())[0]
                materials_info[0] = {
                    'name': 'default_material',
                    'doubleSided': True,
                    'diffuseTexture': default_texture,
                    'diffuseFactor': [1, 1, 1, 1]
                }
        
        result = {
            'gltf_data': gltf_data,
            'textures': textures_info,
            'materials': materials_info,
            'model_folder': model_folder,
            'vertices': vertices,
            'faces': faces,
            'uvs': uvs,
            'colors': colors
        }
        
        print(f"Final result: {len(vertices)} vertices, {len(faces)} faces, {len(uvs)} UVs, {len(textures_info)} textures, {len(materials_info)} materials")
        return result
    except Exception as e:
        st.error(f"Error loading GLTF with textures: {e}")
        return None

#def create_threejs_viewer_with_textures_DISABLED(gltf_info, height=600):
#    """Create Three.js viewer HTML using proper GLTFLoader."""
#    
#    model_folder = gltf_info['model_folder']
#    
#    # Prepare mesh data for JavaScript
#    mesh_data_js = {
#        'vertices': gltf_info['vertices'].tolist() if len(gltf_info['vertices']) > 0 else [],
#        'faces': gltf_info['faces'].tolist() if len(gltf_info['faces']) > 0 else [],
#        'uvs': gltf_info['uvs'].tolist() if len(gltf_info['uvs']) > 0 else [],
#        'colors': gltf_info['colors'].tolist() if len(gltf_info['colors']) > 0 else []
#    }
#    
#    # Convert to JSON string safely
#    mesh_data_json = json.dumps(mesh_data_js)
#    
#    # Get the GLTF file path relative to the model folder
#    gltf_file_name = "scene.gltf"  # Default name
#    if os.path.exists(os.path.join(model_folder, "scene.gltf")):
#        gltf_file_name = "scene.gltf"
#    elif os.path.exists(os.path.join(model_folder, "scene.glb")):
#        gltf_file_name = "scene.glb"
#    
#    # Create a data URL for the GLTF file
#    gltf_path = os.path.join(model_folder, gltf_file_name)
#    with open(gltf_path, 'rb') as f:
#        gltf_data_bytes = f.read()
#    
#    gltf_base64 = base64.b64encode(gltf_data_bytes).decode('utf-8')
#    gltf_data_url = f"data:model/gltf-binary;base64,{gltf_base64}"
#    
#    viewer_html = f"""
#    <!DOCTYPE html>
#    <html>
#    <head>
#        <style>
#            body {{ margin: 0; padding: 0; overflow: hidden; }}
#            #canvas {{ width: 100%; height: {height}px; }}
#            #controls {{
#                position: absolute;
#                top: 10px;
#                left: 10px;
#                background: rgba(0,0,0,0.7);
#                color: white;
#                padding: 10px;
#                border-radius: 5px;
#                font-family: Arial, sans-serif;
#                font-size: 12px;
#                z-index: 1000;
#            }}
#            #loading {{
#                position: absolute;
#                top: 50%;
#                left: 50%;
#                transform: translate(-50%, -50%);
#                color: white;
#                font-size: 18px;
#                z-index: 1001;
#            }}
#        </style>
#    </head>
#    <body>
#        <div id="controls">
#            🖱️ Left-click: Rotate | Right-click: Pan | Scroll: Zoom
#        </div>
#        <div id="loading">Loading 3D model...</div>
#        <div id="canvas"></div>
#        
#        <script type="importmap">
#        {{
#            "imports": {{
#                "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
#                "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
#            }}
#        }}
#        </script>
#        
#        <script type="module">
#            try {{
#            import * as THREE from 'three';
#            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
#            import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
#            
#            // Scene setup
#            const scene = new THREE.Scene();
#            scene.background = new THREE.Color(0x000000);
#            
#            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / {height}, 0.1, 1000);
#            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
#            renderer.setSize(window.innerWidth, {height});
#            renderer.outputColorSpace = THREE.SRGBColorSpace;
#            renderer.toneMapping = THREE.ACESFilmicToneMapping;
#            renderer.toneMappingExposure = 1.0;
#            document.getElementById('canvas').appendChild(renderer.domElement);
#            
#            // Lighting setup for PBR materials
#            const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
#            scene.add(ambientLight);
#            
#            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
#            directionalLight.position.set(5, 5, 5);
#            directionalLight.castShadow = true;
#            scene.add(directionalLight);
#            
#            const hemisphereLight = new THREE.HemisphereLight(0x87CEEB, 0x362d1d, 0.3);
#            scene.add(hemisphereLight);
#            
#            // Controls
#            const controls = new OrbitControls(camera, renderer.domElement);
#            controls.enableDamping = true;
#            controls.dampingFactor = 0.05;
#            controls.enableZoom = true;
#            controls.enablePan = true;
#            controls.enableRotate = true;
#            
#            // Load textures
#            const textures = {json.dumps(gltf_info['textures'])};
#            const textureLoader = new THREE.TextureLoader();
#            
#            // Create materials
#            const materialsData = {json.dumps(gltf_info['materials'])};
#            const materials = {{}};
#            
#            // Create Three.js materials from the material data
#            for (const [key, matData] of Object.entries(materialsData)) {{
#                const material = new THREE.MeshStandardMaterial();
#                material.name = matData.name || 'material_' + key;
#                material.side = matData.doubleSided ? THREE.DoubleSide : THREE.FrontSide;
#                
#                // Handle PBR properties
#                if (matData.baseColorFactor) {{
#                    material.color.setRGB(
#                        matData.baseColorFactor[0],
#                        matData.baseColorFactor[1], 
#                        matData.baseColorFactor[2]
#                    );
#                    if (matData.baseColorFactor.length > 3) {{
#                        material.opacity = matData.baseColorFactor[3];
#                        material.transparent = matData.baseColorFactor[3] < 1.0;
#                    }}
#                }}
#                
#                // Handle textures
#                if (matData.baseColorTexture) {{
#                    const texture = textureLoader.load(matData.baseColorTexture);
#                    texture.colorSpace = THREE.SRGBColorSpace;
#                    material.map = texture;
#                }}
#                
#                if (matData.diffuseTexture) {{
#                    const texture = textureLoader.load(matData.diffuseTexture);
#                    texture.colorSpace = THREE.SRGBColorSpace;
#                    material.map = texture;
#                }}
#                
#                materials[key] = material;
#            }}
#            
#            // Debug materials and textures
#            console.log('Available textures:', Object.keys(textures));
#            console.log('Available materials:', Object.keys(materials));
#            if (Object.keys(materials).length > 0) {{
#                console.log('First material:', materials[0]);
#            }}
#            
#            // Load model using trimesh data (more reliable)
#            const model = new THREE.Group();
#            
#            // Get mesh data from trimesh (passed from Python)
#            const meshData = {mesh_data_json};
#            
#            // Debug mesh data
#            console.log('Mesh data loaded:', {
#                vertices: meshData.vertices ? meshData.vertices.length : 0,
#                faces: meshData.faces ? meshData.faces.length : 0,
#                uvs: meshData.uvs ? meshData.uvs.length : 0,
#                colors: meshData.colors ? meshData.colors.length : 0
#            });
#            
#            // Validate mesh data
#            if (!meshData.vertices || meshData.vertices.length === 0) {
#                console.error('No vertex data available');
#                document.getElementById('loading').innerHTML = 'Error: No vertex data found in model';
#                return;
#            }
#            
#            if (meshData.vertices && meshData.vertices.length > 0) {
#                console.log('Creating geometry from mesh data...');
#                console.log('Vertices count:', meshData.vertices.length);
#                console.log('Faces count:', meshData.faces ? meshData.faces.length : 0);
#                
#                // Create geometry from mesh data
#                const geometry = new THREE.BufferGeometry();
#                
#                // Set vertices
#                const vertices = new Float32Array(meshData.vertices.flat());
#                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
#                console.log('Vertices set:', vertices.length / 3, 'vertices');
#                
#                // Handle faces (convert to triangles if needed)
#                let triangles = [];
#                if (meshData.faces && meshData.faces.length > 0) {
#                    console.log('Processing', meshData.faces.length, 'faces...');
#                    for (let face of meshData.faces) {
#                        if (face.length === 3) {
#                            triangles.push(face[0], face[1], face[2]);
#                        } else if (face.length === 4) {
#                            // Convert quad to two triangles
#                            triangles.push(face[0], face[1], face[2]);
#                            triangles.push(face[0], face[2], face[3]);
#                        }
#                    }
#                    geometry.setIndex(triangles);
#                    console.log('Faces processed:', triangles.length / 3, 'triangles');
#                } else {
#                    console.log('No faces data available, creating non-indexed geometry');
#                }
#                
#                // Set UV coordinates if available, or generate basic UVs
#                if (meshData.uvs && meshData.uvs.length > 0) {
#                    const uvs = new Float32Array(meshData.uvs.flat());
#                    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
#                    console.log('UV coordinates set:', uvs.length / 2, 'UV pairs');
#                } else {
#                    console.log('No UV coordinates available, generating basic UVs');
#                    // Generate basic UV coordinates based on vertex positions
#                    const positionAttribute = geometry.getAttribute('position');
#                    const uvCount = positionAttribute.count;
#                    const uvs = new Float32Array(uvCount * 2);
#                    
#                    for (let i = 0; i < uvCount; i++) {
#                        const x = positionAttribute.getX(i);
#                        const y = positionAttribute.getY(i);
#                        const z = positionAttribute.getZ(i);
#                        
#                        // Simple planar mapping based on X and Z coordinates
#                        const u = (x - geometry.boundingBox.min.x) / (geometry.boundingBox.max.x - geometry.boundingBox.min.x);
#                        const v = (z - geometry.boundingBox.min.z) / (geometry.boundingBox.max.z - geometry.boundingBox.min.z);
#                        
#                        uvs[i * 2] = isNaN(u) ? 0 : u;
#                        uvs[i * 2 + 1] = isNaN(v) ? 0 : v;
#                    }
#                    
#                    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
#                    console.log('Generated UV coordinates:', uvs.length / 2, 'UV pairs');
#                }
#                
#                // Compute normals if not present
#                if (!geometry.attributes.normal) {
#                    geometry.computeVertexNormals();
#                    console.log('Computed vertex normals');
#                }
#                
#                // Compute bounding box for UV generation
#                geometry.computeBoundingBox();
#                
#                // Create material (use first available material or default)
#                const materialIndex = 0;
#                let material = materials[materialIndex];
#                
#                // If no material found, create a default one
#                if (!material) {
#                    material = new THREE.MeshStandardMaterial({
#                        color: 0x888888,
#                        side: THREE.DoubleSide
#                    });
#                    console.log('No material found, using default');
#                }
#                
#                // If material exists but has no texture, try to apply any available texture
#                if (material && !material.map && Object.keys(textures).length > 0) {
#                    const firstTexture = Object.values(textures)[0];
#                    const texture = textureLoader.load(firstTexture);
#                    texture.colorSpace = THREE.SRGBColorSpace;
#                    material.map = texture;
#                    console.log('Applied fallback texture to material');
#                }
#                
#                // Debug material
#                console.log('Using material:', material);
#                console.log('Material properties:', {
#                    name: material.name,
#                    map: material.map ? 'texture loaded' : 'no texture',
#                    color: material.color,
#                    side: material.side
#                });
#                
#                // Create mesh
#                const mesh = new THREE.Mesh(geometry, material);
#                console.log('Mesh created with geometry:', geometry);
#                console.log('Mesh bounding box:', geometry.boundingBox);
#                console.log('Mesh vertices count:', geometry.attributes.position.count);
#                
#                model.add(mesh);
#                console.log('Mesh added to model group');
#            } else {
#                // Fallback: create a simple box
#                const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
#                const material = new THREE.MeshStandardMaterial({ color: 0x888888 });
#                const mesh = new THREE.Mesh(boxGeometry, material);
#                model.add(mesh);
#            }
#            
#            scene.add(model);
#            
#            // Center and scale the model
#            const box = new THREE.Box3().setFromObject(model);
#            const center = box.getCenter(new THREE.Vector3());
#            const size = box.getSize(new THREE.Vector3());
#            const maxDim = Math.max(size.x, size.y, size.z);
#            const scale = 5 / maxDim;
#            
#            console.log('Model bounds:', {
#                center: center,
#                size: size,
#                maxDimension: maxDim,
#                scale: scale
#            });
#            
#            model.scale.set(scale, scale, scale);
#            model.position.sub(center.multiplyScalar(scale));
#            
#            console.log('Model positioned at:', model.position);
#            console.log('Model scale:', model.scale);
#            
#            // Position camera
#            camera.position.set(0, 2, 8);
#            camera.lookAt(0, 0, 0);
#            controls.update();
#            
#            console.log('Camera positioned at:', camera.position);
#            console.log('Camera looking at:', new THREE.Vector3(0, 0, 0));
#            
#            // Hide loading message
#            document.getElementById('loading').style.display = 'none';
#            console.log('Loading message hidden, model should be visible');
#            
#            // Animation loop
#            function animate() {
#                requestAnimationFrame(animate);
#                controls.update();
#                renderer.render(scene, camera);
#            }
#            animate();
#            console.log('Animation loop started');
#            
#            // Handle resize
#            window.addEventListener('resize', function() {
#                camera.aspect = window.innerWidth / {height};
#                camera.updateProjectionMatrix();
#                renderer.setSize(window.innerWidth, {height});
#            });
#            
#            }} catch (error) {{
#                console.error('Error in Three.js viewer:', error);
#                document.getElementById('loading').innerHTML = 'Error loading model: ' + error.message;
#            }}
#            }} catch (error) {{
#                console.error('Error in module import or script:', error);
#                document.getElementById('loading').innerHTML = 'Error loading script: ' + error.message;
#            }}
#        </script>
#    </body>
#    </html>
#    """
#    
#    return viewer_html

def create_simple_textured_viewer(gltf_info, height=600):
    """Create a simple Three.js viewer with texture support."""
    
    # Prepare mesh data for JavaScript
    mesh_data_js = {
        'vertices': gltf_info['vertices'].tolist() if len(gltf_info['vertices']) > 0 else [],
        'faces': gltf_info['faces'].tolist() if len(gltf_info['faces']) > 0 else [],
        'uvs': gltf_info['uvs'].tolist() if len(gltf_info['uvs']) > 0 else [],
        'colors': gltf_info['colors'].tolist() if len(gltf_info['colors']) > 0 else []
    }
    
    # Convert to JSON string safely
    mesh_data_json = json.dumps(mesh_data_js)
    
    # Find texture data
    texture_data = None
    if gltf_info['textures']:
        texture_data = list(gltf_info['textures'].values())[0]  # Use first texture
    
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
            document.getElementById('canvas').appendChild(renderer.domElement);
            
            // Enhanced lighting for brighter display
            const ambientLight = new THREE.AmbientLight(0xffffff, 1.2);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 5, 5);
            scene.add(directionalLight);
            
            // Controls
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Load mesh data
            const meshData = {mesh_data_json};
            
            console.log('Mesh data:', meshData.vertices.length, 'vertices,', meshData.faces.length, 'faces');
            
            // Create geometry
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(meshData.vertices.flat(), 3));
            
            // Handle faces
            let triangles = [];
            for (let face of meshData.faces) {{
                if (face.length === 3) {{
                    triangles.push(face[0], face[1], face[2]);
                }} else if (face.length === 4) {{
                    triangles.push(face[0], face[1], face[2]);
                    triangles.push(face[0], face[2], face[3]);
                }}
            }}
            geometry.setIndex(triangles);
            
            // Generate UV coordinates if needed
            if (meshData.uvs.length === 0) {{
                console.log('Generating UV coordinates...');
                const positionAttribute = geometry.getAttribute('position');
                const uvCount = positionAttribute.count;
                const uvs = new Float32Array(uvCount * 2);
                
                for (let i = 0; i < uvCount; i++) {{
                    const x = positionAttribute.getX(i);
                    const z = positionAttribute.getZ(i);
                    uvs[i * 2] = (x + 100) / 200;  // Simple normalization
                    uvs[i * 2 + 1] = (z + 100) / 200;
                }}
                geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
            }} else {{
                geometry.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(meshData.uvs.flat()), 2));
            }}
            
            // Compute normals
            geometry.computeVertexNormals();
            
            // Create material
            let material;
            if ('{texture_data}') {{
                const textureLoader = new THREE.TextureLoader();
                const texture = textureLoader.load('{texture_data}');
                texture.colorSpace = THREE.SRGBColorSpace;
                material = new THREE.MeshStandardMaterial({{ map: texture, side: THREE.DoubleSide }});
                console.log('Using textured material');
            }} else {{
                material = new THREE.MeshStandardMaterial({{ color: 0xff6b6b, side: THREE.DoubleSide }});
                console.log('Using solid color material');
            }}
            
            // Create mesh
            const mesh = new THREE.Mesh(geometry, material);
            scene.add(mesh);
            
            // Center and scale
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
        print(f"Loading GLTF file: {selected_file}")
        gltf_info = load_gltf_with_textures(selected_file)
        if gltf_info:
            print("GLTF loaded successfully, creating viewer...")
            # Create viewer with texture support
            viewer_html = create_simple_textured_viewer(gltf_info, height=600)
            components.html(viewer_html, height=650)
        else:
            st.error("Failed to load GLTF file")
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
