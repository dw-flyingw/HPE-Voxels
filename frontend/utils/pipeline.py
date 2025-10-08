#!/usr/bin/env python3
"""
pipeline.py

Complete pipeline to convert medical imaging files (.nii.gz) to 3D models with UV masks.
Automates the entire workflow from NIfTI → OBJ → UV Unwrap → Model directories → UV masks.

This pipeline orchestrates all the utilities in frontend/utils:
1. nifti2obj.py - Convert NIfTI to OBJ meshes
2. add_uv_unwrap.py - Add UV coordinates to meshes (xatlas or spherical)
3. obj2model.py - Convert OBJ directly to model directories (GLTF/GLB)
4. create_uv_mask.py - Generate UV masks for FLUX.1

Requirements
------------
All dependencies from frontend/requirements.txt

Example
-------
    # Process all files with default settings
    python pipeline.py
    
    # Custom input/output directories
    python pipeline.py -i ./my_nifti_files -o ./my_output
    
    # With custom processing parameters
    python pipeline.py --smoothing 15 --decimation 0.3 --mask-size 2048
    
    # Skip certain steps
    python pipeline.py --skip-masks --verbose
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List
import shutil


class Pipeline:
    """Medical imaging to 3D model pipeline."""
    
    def __init__(
        self,
        input_dir: str = "input/nifti",
        output_base: str = "output",
        verbose: bool = False,
        overwrite: bool = False
    ):
        self.input_dir = Path(input_dir)
        self.output_base = Path(output_base)
        self.verbose = verbose
        self.overwrite = overwrite
        
        # Define output directories
        self.obj_dir = self.output_base / "obj"
        self.models_dir = self.output_base / "models"
        
        # Get script directory
        self.utils_dir = Path(__file__).parent
        self.project_root = self.utils_dir.parent.parent
        
    def print_step(self, step: int, total: int, message: str):
        """Print a formatted step message."""
        print(f"\n{'='*70}")
        print(f"STEP {step}/{total}: {message}")
        print(f"{'='*70}\n")
    
    def run_command(self, cmd: List[str], step_name: str) -> bool:
        """Run a command and handle errors."""
        try:
            if self.verbose:
                print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=not self.verbose,
                text=True,
                cwd=self.project_root
            )
            
            if not self.verbose and result.stdout:
                print(result.stdout)
            
            print(f"✓ {step_name} completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error in {step_name}:", file=sys.stderr)
            if e.stdout:
                print(e.stdout, file=sys.stderr)
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            return False
        except Exception as e:
            print(f"✗ Unexpected error in {step_name}: {e}", file=sys.stderr)
            return False
    
    def check_input_files(self) -> bool:
        """Check if input directory exists and contains .nii.gz files."""
        if not self.input_dir.exists():
            print(f"✗ Error: Input directory '{self.input_dir}' does not exist")
            return False
        
        nifti_files = list(self.input_dir.glob("*.nii.gz"))
        if not nifti_files:
            print(f"✗ Error: No .nii.gz files found in '{self.input_dir}'")
            return False
        
        print(f"✓ Found {len(nifti_files)} NIfTI file(s) to process:")
        for f in nifti_files:
            print(f"  - {f.name}")
        
        return True
    
    def step_nifti_to_obj(
        self,
        threshold: float = 0.1,
        smoothing: int = 10,
        decimation: float = 0.5,
        close_boundaries: bool = False
    ) -> bool:
        """Step 1: Convert NIfTI files to OBJ meshes."""
        self.print_step(1, 5, "Converting NIfTI to OBJ meshes")
        
        cmd = [
            "python",
            str(self.utils_dir / "nifti2obj.py"),
            "-i", str(self.input_dir),
            "-o", str(self.obj_dir),
            "-t", str(threshold),
            "-s", str(smoothing),
            "-d", str(decimation),
        ]
        
        if close_boundaries:
            cmd.append("--close-boundaries")
        
        if self.verbose:
            cmd.append("-v")
        
        return self.run_command(cmd, "NIfTI → OBJ conversion")
    
    def step_add_uv_unwrap(self, method: str = 'smart') -> bool:
        """Step 2: Add UV coordinates to OBJ files."""
        self.print_step(2, 5, "Adding UV coordinates to OBJ meshes")
        
        cmd = [
            "python",
            str(self.utils_dir / "add_uv_unwrap.py"),
            "-i", str(self.obj_dir),
            "--in-place",
            "-m", method,
            "--overwrite"
        ]
        
        if self.verbose:
            cmd.append("-v")
        
        return self.run_command(cmd, "UV unwrapping")
    
    def step_obj_to_models(self) -> bool:
        """Step 3: Convert OBJ files directly to model directories."""
        self.print_step(3, 5, "Converting OBJ to model directories")
        
        cmd = [
            "python",
            str(self.utils_dir / "obj2model.py"),
            "-i", str(self.obj_dir),
            "-o", str(self.models_dir),
        ]
        
        if self.verbose:
            cmd.append("-v")
        
        return self.run_command(cmd, "OBJ → Models conversion")
    
    def step_create_uv_masks(self, mask_size: int = 1024, create_variants: bool = False) -> bool:
        """Step 4: Create UV masks for FLUX.1 texture generation."""
        self.print_step(4, 5, "Creating UV masks for FLUX.1")
        
        cmd = [
            "python",
            str(self.utils_dir / "create_uv_mask.py"),
            "--models-dir", str(self.models_dir),
            "--size", str(mask_size),
        ]
        
        if self.overwrite:
            cmd.append("--overwrite")
        
        if create_variants:
            cmd.append("--variants")
        
        return self.run_command(cmd, "UV mask creation")
    
    def step_summary(self) -> bool:
        """Step 5: Display summary of created files."""
        self.print_step(5, 5, "Pipeline Summary")
        
        # Count files
        obj_files = list(self.obj_dir.glob("*.obj")) if self.obj_dir.exists() else []
        model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()] if self.models_dir.exists() else []
        uv_masks = list(self.models_dir.glob("*/uv_mask.png")) if self.models_dir.exists() else []
        
        print(f"✓ Pipeline completed successfully!\n")
        print(f"Output Summary:")
        print(f"  - OBJ files:      {len(obj_files):3d} → {self.obj_dir}")
        print(f"  - Model dirs:     {len(model_dirs):3d} → {self.models_dir}")
        print(f"  - UV masks:       {len(uv_masks):3d} → {self.models_dir}/*/uv_mask.png")
        
        if model_dirs:
            print(f"\nCreated model directories:")
            for model_dir in sorted(model_dirs):
                has_mask = (model_dir / "uv_mask.png").exists()
                has_texture = (model_dir / "textures" / "diffuse.png").exists()
                mask_icon = "🎭" if has_mask else "  "
                texture_icon = "🎨" if has_texture else "  "
                print(f"  {mask_icon} {texture_icon} {model_dir.name}")
            print(f"\n  🎭 = UV mask ready  🎨 = Texture available")
        
        print(f"\nNext steps:")
        print(f"  1. Use UV masks with FLUX.1 to generate textures")
        print(f"  2. View models with: python frontend/model_viewer.py")
        print(f"  3. See frontend/utils/README.md for more options")
        
        return True
    
    def run(
        self,
        skip_nifti: bool = False,
        skip_uv_unwrap: bool = False,
        skip_obj: bool = False,
        skip_masks: bool = False,
        nifti_params: Optional[dict] = None,
        uv_params: Optional[dict] = None,
        mask_params: Optional[dict] = None
    ) -> bool:
        """Run the complete pipeline."""
        
        print("╔" + "═"*68 + "╗")
        print("║" + " "*15 + "Medical Imaging Pipeline" + " "*29 + "║")
        print("║" + " "*12 + "NIfTI → OBJ → UV Unwrap → Models → Masks" + " "*12 + "║")
        print("╚" + "═"*68 + "╝")
        
        # Check input files
        if not skip_nifti and not self.check_input_files():
            return False
        
        # Step 1: NIfTI to OBJ
        if not skip_nifti:
            params = nifti_params or {}
            if not self.step_nifti_to_obj(**params):
                return False
        else:
            print("\n[Skipped] Step 1: NIfTI → OBJ conversion")
        
        # Step 2: Add UV Unwrap
        if not skip_uv_unwrap:
            params = uv_params or {}
            if not self.step_add_uv_unwrap(**params):
                return False
        else:
            print("\n[Skipped] Step 2: UV unwrapping")
        
        # Step 3: OBJ to Models (streamlined - no intermediate GLB)
        if not skip_obj:
            if not self.step_obj_to_models():
                return False
        else:
            print("\n[Skipped] Step 3: OBJ → Models conversion")
        
        # Step 4: Create UV Masks
        if not skip_masks:
            params = mask_params or {}
            if not self.step_create_uv_masks(**params):
                return False
        else:
            print("\n[Skipped] Step 4: UV mask creation")
        
        # Step 5: Summary
        self.step_summary()
        
        return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete pipeline: NIfTI → OBJ → GLB → Models → UV Masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with defaults
  python pipeline.py
  
  # Custom input/output
  python pipeline.py -i ./data/nifti -o ./data/output
  
  # High quality settings
  python pipeline.py --smoothing 20 --decimation 0.2 --mask-size 2048
  
  # Skip certain steps (useful for re-runs)
  python pipeline.py --skip-nifti --skip-obj --overwrite
  
  # Only create UV masks for existing models
  python pipeline.py --skip-nifti --skip-obj --skip-glb --overwrite
        """
    )
    
    # Input/Output
    parser.add_argument(
        "-i", "--input",
        default="input/nifti",
        help="Input directory with .nii.gz files (default: input/nifti)"
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output base directory (default: output)"
    )
    
    # NIfTI conversion parameters
    nifti_group = parser.add_argument_group("NIfTI conversion options")
    nifti_group.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.1,
        help="Iso-surface threshold (default: 0.1)"
    )
    nifti_group.add_argument(
        "-s", "--smoothing",
        type=int,
        default=10,
        help="Smoothing iterations (default: 10)"
    )
    nifti_group.add_argument(
        "-d", "--decimation",
        type=float,
        default=0.5,
        help="Mesh decimation 0.0-1.0 (default: 0.5)"
    )
    nifti_group.add_argument(
        "--close-boundaries",
        action="store_true",
        help="Close mesh boundaries to make watertight"
    )
    
    # UV unwrapping parameters
    uv_group = parser.add_argument_group("UV unwrapping options")
    uv_group.add_argument(
        "--uv-method",
        choices=['smart', 'xatlas', 'spherical', 'cylindrical'],
        default='smart',
        help="UV mapping method (default: smart - tries xatlas, falls back to spherical)"
    )
    
    # UV mask parameters
    mask_group = parser.add_argument_group("UV mask options")
    mask_group.add_argument(
        "--mask-size",
        type=int,
        default=1024,
        help="UV mask size in pixels (default: 1024)"
    )
    mask_group.add_argument(
        "--mask-variants",
        action="store_true",
        help="Create all UV mask variants (binary, soft, filled)"
    )
    
    # Pipeline control
    control_group = parser.add_argument_group("Pipeline control")
    control_group.add_argument(
        "--skip-nifti",
        action="store_true",
        help="Skip NIfTI → OBJ conversion"
    )
    control_group.add_argument(
        "--skip-uv-unwrap",
        action="store_true",
        help="Skip UV unwrapping step"
    )
    control_group.add_argument(
        "--skip-obj",
        action="store_true",
        help="Skip OBJ → Models conversion"
    )
    control_group.add_argument(
        "--skip-masks",
        action="store_true",
        help="Skip UV mask creation"
    )
    
    # General options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create pipeline
    pipeline = Pipeline(
        input_dir=args.input,
        output_base=args.output,
        verbose=args.verbose,
        overwrite=args.overwrite
    )
    
    # NIfTI conversion parameters
    nifti_params = {
        'threshold': args.threshold,
        'smoothing': args.smoothing,
        'decimation': args.decimation,
        'close_boundaries': args.close_boundaries
    }
    
    # UV unwrapping parameters
    uv_params = {
        'method': args.uv_method
    }
    
    # UV mask parameters
    mask_params = {
        'mask_size': args.mask_size,
        'create_variants': args.mask_variants
    }
    
    # Run pipeline
    success = pipeline.run(
        skip_nifti=args.skip_nifti,
        skip_uv_unwrap=args.skip_uv_unwrap,
        skip_obj=args.skip_obj,
        skip_masks=args.skip_masks,
        nifti_params=nifti_params,
        uv_params=uv_params,
        mask_params=mask_params
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

