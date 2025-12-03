import glob
import os

def consolidate_files():
    output_file = "all_code.md"
    py_files = glob.glob("*.py")
    
    # Sort for consistent order
    py_files.sort()
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("# Consolidated Python Code\n\n")
        
        for py_file in py_files:
            if py_file == "consolidate_code.py":
                continue
                
            outfile.write(f"## {py_file}\n\n")
            outfile.write("```python\n")
            
            try:
                with open(py_file, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"# Error reading file: {e}")
                
            outfile.write("\n```\n\n")
            
    print(f"Successfully consolidated {len(py_files)} files into {output_file}")

if __name__ == "__main__":
    consolidate_files()
