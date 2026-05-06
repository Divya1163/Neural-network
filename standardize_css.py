import os
import re

css_dir = r"c:/Users/sjai5/OneDrive/ドキュメント/deep_learning_toolbox/static/css"

# Mapping of hardcoded un-professional colors to the standard design system logic variables
replacements = {
    # Whites and grays (backgrounds)
    r'#ffffff': 'var(--bg-primary)',
    r'#f8fafc': 'var(--bg-secondary)',
    r'#fcfefd': 'var(--bg-secondary)',
    r'#f8fbfa': 'var(--bg-secondary)',
    r'#f8fbff': 'var(--bg-secondary)',
    r'#fefe[A-Fa-f0-9]{2}': 'var(--bg-primary)',
    r'#f7fbfa': 'var(--bg-secondary)',
    
    # Texts
    r'#1f2937': 'var(--text-primary)',
    r'#111827': 'var(--text-primary)',
    r'#374151': 'var(--text-primary)',
    r'#0f172a': 'var(--text-primary)',
    r'#1e3a8a': 'var(--text-primary)',
    
    r'#475569': 'var(--text-secondary)',
    r'#64748b': 'var(--text-secondary)',
    r'#334155': 'var(--text-secondary)',
    
    # Borders
    r'#e2e8f0': 'var(--border-color)',
    r'#cbd5e1': 'var(--border-color)',
    r'#dbe8e2': 'var(--border-color)',
    r'#dcefe8': 'var(--border-color)',
    r'#dbeafe': 'var(--border-color)',
    r'#dbe3ef': 'var(--border-color)',
    r'#cfe1ff': 'var(--border-color)',
    r'#888888': 'var(--border-color)',
    r'#888': 'var(--border-color)',
    
    # Blues to Greens (since primary is green)
    r'#1e40af': 'var(--primary-green-dark)',
    r'#2563eb': 'var(--primary-green)',
    r'#1d4ed8': 'var(--primary-green-dark)',
    r'#60a5fa': 'var(--primary-green-light)',
    
    # Accents (light blues/greens)
    r'#eff6ff': 'rgba(5, 150, 105, 0.05)',
    r'#eaf2ff': 'rgba(5, 150, 105, 0.05)',
    r'#93c5fd': 'rgba(5, 150, 105, 0.2)',
    r'#bfdbfe': 'rgba(5, 150, 105, 0.2)',
    
    # Font Families - remove or replace with variable
    r"font-family: 'Manrope', 'Segoe UI', sans-serif;": "",
    r"font-family: 'Manrope', 'sans-serif';": "",
    
    # Specific padding/margin unified overrides
    # Find any box-shadow that is generic and replace with var
    r'box-shadow: 0 4px 12px rgba\(0, 0, 0, 0\.05\)': 'box-shadow: var(--shadow-sm)',
    r'box-shadow: 0 12px 24px rgba\(0, 0, 0, 0\.1\)': 'box-shadow: var(--shadow-lg)',
    r'box-shadow: 0 4px 14px rgba\(15, 23, 42, 0\.04\)': 'box-shadow: var(--shadow-md)',
    
    # Replace any border-radius: 1rem with 0.625rem to match model cards
    r'border-radius:\s*1rem': 'border-radius: 0.625rem',
    r'border-radius:\s*0\.75rem': 'border-radius: 0.625rem',
    r'border-radius:\s*0\.7rem': 'border-radius: 0.625rem',
    
    # Replace cards padding to exactly match dashboard
    r'padding:\s*1\.2rem': 'padding: 1.5rem',
    r'padding:\s*1rem 1rem 1\.05rem': 'padding: 1.5rem',
}

for filename in os.listdir(css_dir):
    if not filename.endswith('.css') or filename == 'style.css':
        continue
        
    filepath = os.path.join(css_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    original = content
    for pattern, repl in replacements.items():
        # Using regex replace with ignoring case for hex colors where applicable
        if pattern.startswith('#'):
            content = re.sub(pattern, repl, content, flags=re.IGNORECASE)
        else:
            content = re.sub(pattern, repl, content)
            
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filename}")
