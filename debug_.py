# ============================================================================
# Debug Script: Check Project Structure and Fix Import Issues
# ============================================================================

import sys
import os
from pathlib import Path

print("🔍 DEBUGGING PROJECT STRUCTURE")
print("=" * 50)

# 1. Check current working directory
print(f"Current working directory: {Path.cwd()}")
print(f"Current directory name: {Path.cwd().name}")

# 2. Determine project root
if Path.cwd().name == "notebooks":
    project_root = Path.cwd().parent
    print(f"Detected notebook environment")
else:
    project_root = Path.cwd()
    print(f"Detected root environment")

print(f"Project root: {project_root}")

# 3. Check src directory
src_path = project_root / "src"
print(f"Expected src path: {src_path}")
print(f"Src directory exists: {src_path.exists()}")

if src_path.exists():
    print(f"Contents of src directory:")
    for item in src_path.iterdir():
        if item.is_file():
            print(f"  📄 {item.name}")
        elif item.is_dir():
            print(f"  📁 {item.name}/")
            # Check for __init__.py files
            init_file = item / "__init__.py"
            if init_file.exists():
                print(f"    ✅ Has __init__.py")
            else:
                print(f"    ❌ Missing __init__.py")
else:
    print("❌ Src directory not found!")
    print("Contents of project root:")
    for item in project_root.iterdir():
        if item.is_file():
            print(f"  📄 {item.name}")
        elif item.is_dir():
            print(f"  📁 {item.name}/")

# 4. Check Python path
print(f"\nCurrent Python path:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

# 5. Try to add src to path and test imports
print(f"\n🔧 ATTEMPTING TO FIX IMPORTS")
print("-" * 30)

if src_path.exists():
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Added {src_path} to Python path")
    
    # Test each module import
    modules_to_test = ['config', 'core', 'models', 'evaluation', 'visualization']
    
    for module_name in modules_to_test:
        module_path = src_path / module_name
        print(f"\nTesting module: {module_name}")
        print(f"  Expected path: {module_path}")
        print(f"  Exists as directory: {module_path.is_dir()}")
        print(f"  Exists as file: {(src_path / f'{module_name}.py').exists()}")
        
        if module_path.is_dir():
            init_file = module_path / "__init__.py"
            print(f"  Has __init__.py: {init_file.exists()}")
            if init_file.exists():
                print(f"  Contents of __init__.py:")
                try:
                    with open(init_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            print(f"    {content[:200]}...")
                        else:
                            print(f"    (empty file)")
                except Exception as e:
                    print(f"    Error reading file: {e}")
        
        # Try importing
        try:
            __import__(module_name)
            print(f"  ✅ Import successful")
        except ImportError as e:
            print(f"  ❌ Import failed: {e}")
        except Exception as e:
            print(f"  ⚠️ Other error: {e}")

print(f"\n💡 RECOMMENDATIONS:")
print("-" * 20)

if not src_path.exists():
    print("1. Create the 'src' directory in your project root")
    print("2. Organize your modules inside the src directory")
else:
    print("1. Ensure each module directory has an __init__.py file")
    print("2. Check that your module files contain the expected classes/functions")
    print("3. Make sure there are no syntax errors in your module files")

print("\n🔧 Next steps:")
print("1. Run this debug script first")
print("2. Fix any missing files/directories it identifies")
print("3. Then try your original notebook again")