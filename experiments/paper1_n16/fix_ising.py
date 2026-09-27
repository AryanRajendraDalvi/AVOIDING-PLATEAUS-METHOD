import re
import glob

def fix_ising():
    files = glob.glob('*.py')
    pattern = re.compile(r"qml\.IsingZZ\(([^,]+),\s*wires=\[([^,]+),\s*([^\]]+)\]\)")
    replacement = r"qml.CNOT(wires=[\2, \3]); qml.RZ(\1, wires=\3); qml.CNOT(wires=[\2, \3])"
    
    for f in files:
        if f == 'fix_ising.py':
            continue
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        
        new_content, count = pattern.subn(replacement, content)
        new_content = new_content.replace('diff_method="backprop"', 'diff_method="adjoint"')
        new_content = new_content.replace('qml.device("default.qubit"', 'qml.device("lightning.qubit"')
        
        if count > 0 or "adjoint" in new_content or "lightning.qubit" in new_content:
            print(f"Fixed {f}")
            with open(f, 'w', encoding='utf-8') as file:
                file.write(new_content)

if __name__ == "__main__":
    fix_ising()
