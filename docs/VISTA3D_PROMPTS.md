# Vista3D Anatomical Structure Prompts

This document shows examples of the hyper photo-realistic prompts automatically generated for Nvidia Vista3D anatomical structures.

## Overview

The Model Viewer automatically generates medically accurate, hyper photo-realistic prompts for all 130+ Vista3D anatomical structures. Each prompt includes:

- **Anatomically correct coloration** - Based on actual tissue appearance
- **Specific texture descriptions** - Accurate surface characteristics
- **Medical photography quality** - Clinical imaging standards
- **8K resolution specification** - Maximum detail
- **Professional medical illustration** - Suitable for clinical use

## Example Prompts by Category

### Organs (Soft Tissue)

#### Liver
```
hyper photo-realistic human liver tissue surface, medical photography, anatomically accurate hepatic parenchyma, rich brownish-red color, smooth glossy surface with subtle texture, natural vascular patterns, clinical imaging quality, 8K resolution, professional medical illustration
```

#### Heart
```
hyper photo-realistic human heart muscle surface, medical photography, anatomically accurate cardiac tissue, deep red color, striated muscle texture with coronary vessels, natural myocardial fibers, clinical imaging quality, 8K resolution, professional medical illustration
```

#### Colon
```
hyper photo-realistic human colon tissue surface, medical photography, anatomically accurate colonic mucosa with haustra, pink to reddish color, segmented texture with taeniae coli, natural mucosal folds, clinical imaging quality, 8K resolution, professional medical illustration
```

#### Kidney
```
hyper photo-realistic human kidney tissue surface, medical photography, anatomically accurate renal capsule, reddish-brown color, smooth surface with subtle granular texture, natural vascular markings, clinical imaging quality, 8K resolution, professional medical illustration
```

#### Brain
```
hyper photo-realistic human brain tissue surface, medical photography, anatomically accurate cerebral cortex, grayish-pink color, gyri and sulci texture, natural vascular patterns, clinical imaging quality, 8K resolution, professional medical illustration
```

### Blood Vessels

#### Aorta
```
hyper photo-realistic human aorta blood vessel surface, medical photography, anatomically accurate arterial wall, smooth red to pink color, elastic tissue texture, natural intimal surface with subtle striations, clinical imaging quality, 8K resolution, professional medical illustration
```

#### Iliac Artery
```
hyper photo-realistic human iliac artery surface, medical photography, anatomically accurate arterial wall, smooth red color, elastic tissue texture, natural vascular lumen, clinical imaging quality, 8K resolution, professional medical illustration
```

#### Inferior Vena Cava
```
hyper photo-realistic human vena cava surface, medical photography, anatomically accurate venous wall, dark red to bluish color, smooth surface texture, natural venous lumen, clinical imaging quality, 8K resolution, professional medical illustration
```

#### Portal Vein
```
hyper photo-realistic human portal vein surface, medical photography, anatomically accurate venous wall, dark red to blue color, smooth surface texture, natural venous appearance, clinical imaging quality, 8K resolution, professional medical illustration
```

### Bones

#### Hip
```
hyper photo-realistic human hip bone surface, medical photography, anatomically accurate pelvic bone texture, white to beige color, smooth cortical surface with trabecular patterns, natural bone structure, clinical imaging quality, 8K resolution, professional medical illustration
```

#### Vertebrae
```
hyper photo-realistic human vertebra bone surface, medical photography, anatomically accurate bone texture, white to beige color, porous trabecular bone structure, natural cortical surface, clinical imaging quality, 8K resolution, professional medical illustration
```

#### Ribs
```
hyper photo-realistic human rib bone surface, medical photography, anatomically accurate bone texture, white to beige color, smooth cortical surface with subtle periosteal texture, natural bone appearance, clinical imaging quality, 8K resolution, professional medical illustration
```

#### Skull
```
hyper photo-realistic human skull bone surface, medical photography, anatomically accurate cranial bone texture, white to beige color, smooth to sutured cortical surface, natural bone structure, clinical imaging quality, 8K resolution, professional medical illustration
```

### Muscles

#### Gluteus
```
hyper photo-realistic human gluteus muscle tissue, medical photography, anatomically accurate skeletal muscle, deep red color, striated muscle fiber texture, natural fascial covering, clinical imaging quality, 8K resolution, professional medical illustration
```

#### Iliopsoas
```
hyper photo-realistic human iliopsoas muscle tissue, medical photography, anatomically accurate skeletal muscle, deep red color, striated muscle fiber texture, natural fascial covering, clinical imaging quality, 8K resolution, professional medical illustration
```

### Lungs

#### Lung Tissue
```
hyper photo-realistic human lung tissue surface, medical photography, anatomically accurate pulmonary parenchyma, pink to grayish color, spongy texture with alveolar patterns, natural pleural surface, clinical imaging quality, 8K resolution, professional medical illustration
```

### Airways

#### Trachea
```
hyper photo-realistic human trachea surface, medical photography, anatomically accurate tracheal mucosa, pale pink color, cartilaginous rings texture, natural ciliated epithelium, clinical imaging quality, 8K resolution, professional medical illustration
```

## Prompt Structure

All prompts follow this medical-grade structure:

```
hyper photo-realistic human [STRUCTURE] [tissue/surface/vessel],
medical photography,
anatomically accurate [specific anatomy],
[accurate coloration],
[specific texture description],
natural [characteristic features],
clinical imaging quality,
8K resolution,
professional medical illustration
```

## Key Elements

### 1. Prefix
- **"hyper photo-realistic"** - Highest quality realistic rendering
- **"medical photography"** - Clinical photography standards

### 2. Anatomical Accuracy
- Specific anatomical terminology
- Correct tissue type designation
- Accurate structure names

### 3. Coloration
- **Organs:** Pink, red, brownish-red based on blood supply
- **Vessels:** Red (arteries), dark red/blue (veins)
- **Bones:** White to beige
- **Muscles:** Deep red
- **Airways:** Pale pink

### 4. Texture Details
- Surface characteristics (smooth, rough, striated, etc.)
- Structural patterns (haustra, rugae, folds, etc.)
- Vascular markings
- Tissue-specific features

### 5. Quality Specifications
- **Clinical imaging quality** - Medical-grade standards
- **8K resolution** - Maximum detail level
- **Professional medical illustration** - Suitable for educational/clinical use

## Automatic Name Normalization

The system automatically handles variations:

| Input Model Name | Normalized | Prompt Category |
|-----------------|------------|-----------------|
| `left_kidney` | kidney | Organ |
| `right_kidney` | kidney | Organ |
| `left_iliac_artery` | iliac artery | Blood vessel (artery) |
| `right_hip` | hip | Bone |
| `vertebrae_L5` | vertebrae l5 | Bone (vertebra) |
| `left_lung_upper_lobe` | lung | Lung tissue |
| `colon` | colon | Organ |

## Fallback Logic

If a specific prompt isn't defined, the system uses intelligent fallbacks:

1. **Check for keywords:**
   - "artery" → arterial vessel prompt
   - "vein" → venous vessel prompt
   - "lung" → lung tissue prompt
   - "muscle" → skeletal muscle prompt
   - "vertebra" or "c1-s1" → vertebra prompt

2. **Vista3D label lookup:**
   - Searches frontend/conf/vista3d_prompts.json
   - Generates anatomically accurate prompt for the structure

3. **Ultimate fallback:**
   ```
   hyper photo-realistic human [name] anatomical structure,
   medical photography,
   anatomically accurate surface texture,
   natural clinical appearance,
   high detail,
   8K resolution,
   professional medical illustration
   ```

## Benefits

### For Clinical/Educational Use
- Medically accurate visualizations
- Anatomically correct coloration
- Professional quality suitable for presentations
- Educational material creation

### For Research
- Consistent texture generation across specimens
- Reproducible results with same prompts
- High-resolution outputs (8K quality)
- Clinical imaging standards

### For Development
- Automatic prompt generation - no manual prompt engineering needed
- Covers all Vista3D structures (130+ anatomies)
- Handles naming variations automatically
- Extensible for new structures

## Usage in Model Viewer

1. **Select any Vista3D model** from dropdown
2. **Prompt is auto-generated** based on structure name
3. **Review prompt** in the preview box
4. **Optionally customize** if needed
5. **Generate texture** with one click

Example workflow:
```bash
# Model: "left_iliac_artery"
# Auto-generated prompt:
# "hyper photo-realistic human iliac artery surface, medical photography,
#  anatomically accurate arterial wall, smooth red color, elastic tissue
#  texture, natural vascular lumen, clinical imaging quality, 8K resolution,
#  professional medical illustration"
```

## Supported Vista3D Structures

The system supports all 130+ structures from `frontend/conf/vista3d_prompts.json` including:

**Major Categories:**
- 20+ organs
- 15+ blood vessels
- 40+ bones (including all vertebrae C1-L6, S1, ribs 1-12)
- 10+ muscles
- 5+ lung lobes
- Airways and respiratory structures
- Glands and cartilage
- Pathological structures (tumors, lesions, cysts)

See `frontend/conf/vista3d_prompts.json` for the complete list.

## Technical Implementation

The prompt generation is handled in:
- **File:** `frontend/logic/model_viewer_logic.py`
- **Function:** `generate_texture_prompt(model_name: str)`
- **Supporting:** `normalize_model_name(model_name: str)`
- **Data Sources:** 
  - `frontend/conf/vista3d_prompts.json` - Texture generation prompts
  - `frontend/conf/vista3d_label_colors.json` - Color definitions

---

**Note:** All prompts are designed to work with the Flux.1-dev model for optimal results. The quality specifications (8K, medical photography, clinical imaging) guide the AI to generate anatomically accurate, medically suitable textures.

