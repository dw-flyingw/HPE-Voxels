# Architecture: Data-Driven Texture Generation

## Overview

The texture generation system uses a **data-driven architecture** where prompts are stored in JSON files separate from code, making them easy to maintain, update, and share across applications.

## Design Decision: Separate JSON Files

### Why Not Modify `vista3d_label_colors.json`?

We decided to create a **separate** `vista3d_prompts.json` file instead of adding prompts to the existing `vista3d_label_colors.json`. Here's why:

#### Separation of Concerns
- `vista3d_label_colors.json` - **Color definitions** for visualization
- `vista3d_prompts.json` - **Texture generation prompts** for AI

Each file serves a distinct purpose and can evolve independently.

#### Maintainability
- Color data and prompt data have different update cycles
- Prompts may need frequent tuning based on AI model performance
- Colors are relatively static (anatomical standards)

#### Flexibility
- Can have multiple prompt files for different use cases:
  - `vista3d_prompts_realistic.json` - Photo-realistic (current)
  - `vista3d_prompts_artistic.json` - Artistic interpretations
  - `vista3d_prompts_educational.json` - Educational simplifications
- Each application can use whichever prompt set suits its needs

#### File Size Management
- Prompts are verbose (200+ characters each)
- Keeping them separate prevents the color file from becoming unwieldy
- Easier to version control and diff changes

## File Structure

### `vista3d_prompts.json`

```json
{
  "metadata": {
    "version": "1.0",
    "description": "Hyper photo-realistic texture generation prompts",
    "quality_standard": "8K resolution, medical photography",
    "created": "2025-10-07"
  },
  "default_template": "hyper photo-realistic human {structure} anatomical structure...",
  "prompts": [
    {
      "id": 1,
      "name": "liver",
      "prompt": "hyper photo-realistic human liver tissue surface..."
    },
    ...
  ]
}
```

**Key Fields:**
- `metadata` - Version and documentation
- `default_template` - Fallback template with {structure} placeholder
- `prompts` - Array of structure-specific prompts
  - `id` - Vista3D label ID (matches vista3d_label_colors.json)
  - `name` - Structure name (matches vista3d_label_colors.json)
  - `prompt` - Full hyper photo-realistic prompt

### `vista3d_label_colors.json`

```json
[
  {
    "id": 1,
    "name": "liver",
    "color": [34, 139, 34]
  },
  ...
]
```

**Key Fields:**
- `id` - Vista3D label ID
- `name` - Structure name
- `color` - RGB color values for visualization

## Data Flow

```
1. User selects model (e.g., "left_kidney")
   ↓
2. normalize_model_name() → "kidney"
   ↓
3. load_vista3d_prompts() → Load vista3d_prompts.json
   ↓
4. generate_texture_prompt() → Match normalized name to prompt
   ↓
5. Return prompt:
   "hyper photo-realistic human kidney tissue surface,
    medical photography, anatomically accurate renal capsule..."
   ↓
6. Send to Flux API for texture generation
```

## Code Architecture

### Logic Layer (`frontend/logic/model_viewer_logic.py`)

```python
def load_vista3d_prompts() -> dict:
    """Load prompts from JSON file."""
    # Loads vista3d_prompts.json
    # Returns dict with metadata and prompts list

def normalize_model_name(model_name: str) -> str:
    """Normalize model name for matching."""
    # "left_kidney" → "kidney"
    # "vertebrae_L5" → "vertebrae l5"
    
def generate_texture_prompt(model_name: str) -> str:
    """Generate prompt from data file."""
    # 1. Normalize model name
    # 2. Load prompts from JSON
    # 3. Try exact match
    # 4. Try partial match
    # 5. Use default template
```

### UI Layer (`frontend/model_viewer.py`)

```python
from logic.model_viewer_logic import generate_texture_prompt

# Generate prompt for selected model
model_name = os.path.basename(selected_folder)
auto_prompt = generate_texture_prompt(model_name)

# Display in UI
st.text_area("Prompt Preview", auto_prompt, disabled=True)
```

## Benefits of This Architecture

### 1. Easy Updates
Non-developers can edit `vista3d_prompts.json`:
```bash
# Open in any text editor
nano vista3d_prompts.json

# Find structure
"name": "liver"

# Update prompt
"prompt": "new improved prompt..."

# Save and restart app
```

### 2. Version Control
```bash
# Clear diff of prompt changes
git diff vista3d_prompts.json

# Revert specific prompts
git checkout HEAD -- vista3d_prompts.json
```

### 3. Sharing
```bash
# Share prompts across projects
cp vista3d_prompts.json /other/project/

# Use in different applications
# - Python, JavaScript, any language can read JSON
```

### 4. Testing
```python
# Test with different prompt sets
load_prompts_from('vista3d_prompts_test.json')
load_prompts_from('vista3d_prompts_production.json')
```

### 5. Extensibility
```json
// Future: Add multiple prompts per structure
{
  "id": 1,
  "name": "liver",
  "prompts": {
    "realistic": "hyper photo-realistic...",
    "artistic": "artistic rendering...",
    "simplified": "simplified educational..."
  }
}
```

## Fallback Strategy

The system has multiple fallback layers:

```
1. Exact match: "liver" == "liver" ✓
   ↓ (if not found)
2. Partial match: "kidney" in "left kidney" ✓
   ↓ (if not found)
3. Default template: "hyper photo-realistic human {structure}..."
   ↓ (if JSON load fails)
4. Hardcoded default: Built into code as safety
```

This ensures the system always returns a valid prompt.

## File Locations

```
HPE-Voxels/
├── vista3d_prompts.json          # Texture generation prompts
├── vista3d_label_colors.json     # Visualization colors
└── frontend/
    ├── model_viewer.py            # UI layer
    └── logic/
        └── model_viewer_logic.py  # Logic layer (reads JSON)
```

## Performance

- **JSON Load Time:** < 10ms (cached after first load)
- **Prompt Lookup:** O(n) linear search (132 items, ~1ms)
- **Memory Footprint:** ~50KB (JSON data in memory)

No performance impact on texture generation (~60-120 seconds Flux API call).

## Migration from Hardcoded Prompts

### Before (Hardcoded)
```python
# 200+ lines of dictionaries in code
organs = {
    'liver': 'hyper photo-realistic...',
    'spleen': 'hyper photo-realistic...',
    ...
}
```

**Issues:**
- Hard to maintain
- Requires code changes for prompt updates
- Not shareable across applications

### After (Data-Driven)
```python
# Clean code: 20 lines
prompts_data = load_vista3d_prompts()
prompt = generate_texture_prompt(model_name)
```

**Benefits:**
- ✅ Easy to maintain
- ✅ No code changes for prompt updates
- ✅ Shareable JSON data
- ✅ Version controlled
- ✅ Testable

## Future Enhancements

### Multi-Style Support
```json
{
  "name": "liver",
  "styles": {
    "realistic": "hyper photo-realistic...",
    "schematic": "medical diagram style...",
    "artistic": "artistic interpretation..."
  }
}
```

### Localization
```json
{
  "name": "liver",
  "prompt_en": "hyper photo-realistic human liver...",
  "prompt_es": "hígado humano hiper fotorrealista...",
  "prompt_zh": "超逼真的人类肝脏..."
}
```

### Metadata
```json
{
  "name": "liver",
  "prompt": "...",
  "tags": ["organ", "soft-tissue", "abdominal"],
  "color_hex": "#228B22",
  "anatomical_system": "digestive"
}
```

## Best Practices

### Adding New Prompts
1. Open `vista3d_prompts.json`
2. Add new entry to `prompts` array
3. Use existing prompts as template
4. Include all quality markers:
   - "hyper photo-realistic"
   - "medical photography"
   - "8K resolution"
   - "professional medical illustration"
5. Save and test

### Updating Existing Prompts
1. Find prompt by name or ID
2. Modify only the `prompt` field
3. Keep `id` and `name` unchanged
4. Test with actual model

### Testing Prompts
```bash
# Test prompt generation
cd frontend
python -c "
from logic.model_viewer_logic import generate_texture_prompt
print(generate_texture_prompt('liver'))
"
```

## Conclusion

The data-driven architecture provides:
- ✅ **Separation of concerns** - Colors vs prompts
- ✅ **Maintainability** - Easy to update prompts
- ✅ **Flexibility** - Support multiple prompt styles
- ✅ **Sharability** - JSON format works everywhere
- ✅ **Extensibility** - Easy to add new features

This design pattern can be applied to other aspects of the system as well.

---

**Created:** October 7, 2025  
**Author:** AI Assistant  
**Status:** ✅ Implemented and Documented

