"""
OBJ Texture Generator - Streamlit GUI
Upload OBJ files and generate hyper-realistic textures using the Flux model
"""

import streamlit as st
import sys
from pathlib import Path

# Add logic folder to path
sys.path.append(str(Path(__file__).parent / "logic"))

from obj_texture_generator_logic import (
    upload_obj_file,
    generate_texture_from_prompt,
    apply_texture_to_obj,
    get_flux_server_url,
    check_flux_server_health,
    load_obj_preview
)

# Page config
st.set_page_config(
    page_title="OBJ Texture Generator",
    page_icon="🎨",
    layout="wide"
)

# Title and description
st.title("🎨 OBJ Texture Generator")
st.markdown("""
Upload a 3D model (.obj file) and provide a prompt to generate a hyper-realistic texture using the Flux AI model.
The generated texture will be applied to your model while preserving its volume.
""")

# Initialize session state
if 'texture_generated' not in st.session_state:
    st.session_state.texture_generated = False
if 'obj_path' not in st.session_state:
    st.session_state.obj_path = None
if 'texture_path' not in st.session_state:
    st.session_state.texture_path = None

# Sidebar - Server configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    flux_url = get_flux_server_url()
    st.text_input("Flux Server URL", value=flux_url, disabled=True, key="flux_url_display")
    
    # Check server health
    if st.button("Check Server Status", use_container_width=True):
        with st.spinner("Checking server..."):
            health = check_flux_server_health(flux_url)
            if health.get('status') == 'healthy':
                st.success("✅ Server is running!")
                st.json(health)
            else:
                st.error("❌ Server is not available")
                st.json(health)
    
    st.divider()
    
    st.header("📊 Generation Settings")
    texture_size = st.selectbox(
        "Texture Size",
        options=[512, 1024, 2048],
        index=1,
        help="Higher resolution = better quality but slower generation"
    )
    
    guidance_scale = st.slider(
        "Guidance Scale",
        min_value=1.0,
        max_value=10.0,
        value=3.5,
        step=0.5,
        help="Higher values = more literal prompt interpretation"
    )
    
    num_steps = st.slider(
        "Inference Steps",
        min_value=20,
        max_value=100,
        value=50,
        step=5,
        help="More steps = better quality but slower"
    )
    
    seed = st.number_input(
        "Seed (optional)",
        min_value=-1,
        max_value=999999,
        value=-1,
        help="Use -1 for random, or set a specific seed for reproducibility"
    )

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("📁 Upload Model")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an OBJ file",
        type=['obj'],
        help="Upload a .obj file from your output folder or any 3D model"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        with st.spinner("Processing uploaded file..."):
            obj_path, obj_info = upload_obj_file(uploaded_file)
            st.session_state.obj_path = obj_path
            
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Display file info
            with st.expander("📊 Model Information", expanded=True):
                st.write(f"**Vertices:** {obj_info.get('vertices', 'N/A'):,}")
                st.write(f"**Faces:** {obj_info.get('faces', 'N/A'):,}")
                st.write(f"**Size:** {obj_info.get('size', 'N/A')}")
                
                if obj_info.get('has_normals'):
                    st.write("✓ Has vertex normals")
                if obj_info.get('has_uvs'):
                    st.write("✓ Has UV coordinates")
                else:
                    st.warning("⚠️ No UV coordinates - will generate basic UV mapping")

with col2:
    st.header("✨ Generate Texture")
    
    # Initialize prompt in session state
    if 'prompt_text' not in st.session_state:
        st.session_state.prompt_text = "hyper photo realistic colon tissue with natural surface details, medical grade quality, 4k resolution"
    
    # Prompt input - using key to bind to session state
    prompt = st.text_area(
        "Texture Prompt",
        value=st.session_state.prompt_text,
        height=100,
        help="Describe the texture you want to generate"
    )
    
    # Example prompts
    with st.expander("💡 Example Prompts"):
        if st.button("Realistic Colon", use_container_width=True):
            st.session_state.prompt_text = "hyper photo realistic colon tissue with haustra and natural surface details, medical grade quality, 8k resolution"
            st.rerun()
        
        if st.button("Realistic Heart", use_container_width=True):
            st.session_state.prompt_text = "hyper photo realistic cardiac muscle tissue with blood vessels, medical photography, 8k resolution"
            st.rerun()
        
        if st.button("Realistic Liver", use_container_width=True):
            st.session_state.prompt_text = "hyper photo realistic liver tissue with smooth surface and blood vessels, medical grade, 8k resolution"
            st.rerun()
    
    # Generate button
    if st.button("🎨 Generate Texture", use_container_width=True, type="primary"):
        if st.session_state.obj_path is None:
            st.error("Please upload an OBJ file first!")
        elif not prompt.strip():
            st.error("Please enter a texture prompt!")
        else:
            # Generate texture
            with st.spinner("🎨 Generating texture with Flux model... This may take a minute..."):
                try:
                    texture_path = generate_texture_from_prompt(
                        prompt=prompt,
                        flux_url=flux_url,
                        width=texture_size,
                        height=texture_size,
                        guidance_scale=guidance_scale,
                        num_steps=num_steps,
                        seed=None if seed == -1 else seed
                    )
                    
                    if texture_path:
                        st.session_state.texture_path = texture_path
                        st.success("✅ Texture generated successfully!")
                    else:
                        st.error("❌ Failed to generate texture")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            # Apply texture to OBJ
            if st.session_state.texture_path:
                with st.spinner("Applying texture to OBJ file..."):
                    try:
                        output_obj_path = apply_texture_to_obj(
                            st.session_state.obj_path,
                            st.session_state.texture_path
                        )
                        
                        if output_obj_path:
                            st.session_state.output_obj_path = output_obj_path
                            st.session_state.texture_generated = True
                            st.success("✅ Texture applied successfully!")
                        else:
                            st.error("❌ Failed to apply texture")
                    
                    except Exception as e:
                        st.error(f"❌ Error applying texture: {str(e)}")

# Results section
if st.session_state.texture_generated:
    st.divider()
    st.header("🎉 Results")
    
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.subheader("Generated Texture")
        if st.session_state.texture_path:
            st.image(st.session_state.texture_path, use_container_width=True)
    
    with result_col2:
        st.subheader("Model Preview")
        st.info("3D preview will be available in the next update")
        # TODO: Add 3D model viewer
    
    # Download section
    st.divider()
    st.subheader("📥 Download Files")
    
    download_col1, download_col2, download_col3 = st.columns(3)
    
    with download_col1:
        if st.session_state.get('output_obj_path'):
            with open(st.session_state.output_obj_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download OBJ",
                    data=f,
                    file_name=Path(st.session_state.output_obj_path).name,
                    mime="application/octet-stream",
                    use_container_width=True
                )
    
    with download_col2:
        if st.session_state.texture_path:
            with open(st.session_state.texture_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download Texture",
                    data=f,
                    file_name="texture.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    with download_col3:
        # Check for MTL file
        mtl_path = str(st.session_state.output_obj_path).replace('.obj', '.mtl')
        if Path(mtl_path).exists():
            with open(mtl_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download MTL",
                    data=f,
                    file_name="material.mtl",
                    mime="application/octet-stream",
                    use_container_width=True
                )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Powered by FLUX.1-dev | HPE Voxels Project</p>
</div>
""", unsafe_allow_html=True)

