#!/usr/bin/env python3
"""
glb_viewer.py

A clean and simple Streamlit app for viewing 3D .glb files.

Usage:
    python glb_viewer.py
    # or with custom port:
    GLB_VIEWER_PORT=8502 python glb_viewer.py
"""

import streamlit as st
import streamlit.components.v1 as components
import trimesh
import os
import numpy as np
import glob
import tempfile
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="GLB Viewer - 3D Model Viewer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_glb_viewer(file_path, height=600):
    """Render a GLB file using Three.js viewer embedded in Streamlit."""
    
    try:
        # Load GLB file with trimesh to get mesh data
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
                
                // Enhanced ambient light for brighter display
                const ambientLight = new THREE.AmbientLight(0xffffff, 1.5);
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
        st.error(f"Error rendering GLB file: {e}")

def load_glb_info(file_path):
    """Load GLB file and return mesh information."""
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
                raise ValueError("No valid mesh data found in GLB file")
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
        st.error(f"Error loading GLB file: {e}")
        return None

# --- Main App ---

# Sidebar
with st.sidebar:
    st.title("🎨 GLB Viewer")
    st.markdown("A clean and simple 3D viewer for .glb files")
    
    st.divider()
    st.header("📁 File Selection")
    
    # Source selection
    source_option = st.radio(
        "Choose source:",
        ["Upload File", "Load from Directory"],
        help="Upload a file or select from the output directory"
    )
    
    selected_file = None
    temp_file = None
    
    if source_option == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload GLB file",
            type=['glb'],
            help="Select a .glb file to view"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_file = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_file = temp_file
            
    else:  # Load from Directory
        glb_dir = "./output/glb"
        if os.path.exists(glb_dir):
            glb_files = glob.glob(os.path.join(glb_dir, "*.glb"))
            glb_files.sort()
            
            if glb_files:
                file_options = ["Select a file..."] + [os.path.basename(f) for f in glb_files]
                selected_filename = st.selectbox(
                    "Select GLB file:",
                    file_options,
                    help="Choose from available .glb files"
                )
                
                if selected_filename != "Select a file...":
                    selected_file = os.path.join(glb_dir, selected_filename)
            else:
                st.warning("No .glb files found in ./output/glb/")
        else:
            st.error("Directory ./output/glb/ not found!")

# Main content
if selected_file and os.path.exists(selected_file):
    # Load mesh information (needed for rendering)
    mesh_info = load_glb_info(selected_file)
    
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
        st.info("👆 Upload a GLB file or select one from the directory to get started.")

# Port configuration (if running as main)
if __name__ == "__main__":
    import sys
    port = int(os.environ.get('GLB_VIEWER_PORT', 8501))
    sys.argv.extend(['--server.port', str(port)])
