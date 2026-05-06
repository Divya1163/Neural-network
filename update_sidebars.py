import os
import re

template_dir = r"c:/Users/sjai5/OneDrive/ドキュメント/deep_learning_toolbox/templates"

files = [
    "perceptron.html",
    "backpropogation_dashboard.html",
    "mlp_dashboard.html",
    "activation_functions_dashboard.html",
    "digit_detection_dashboard.html",
    "cnn_dashboard.html",
    "opencv_dashboard.html",
    "rnn_dashboard.html",
    "sentiment_analysis_dashboard.html",
    "perceptron_dashboard.html"
]

with open(os.path.join(template_dir, "index.html"), "r", encoding="utf-8") as f:
    index_content = f.read()

nav_match = re.search(r'<nav class="sidebar-menu">.*?</nav>', index_content, re.DOTALL)
if not nav_match:
    print("Could not find nav in index.html")
    exit(1)

base_nav = nav_match.group(0)

# remove active class from base_nav
base_nav = base_nav.replace('class="menu-item active"', 'class="menu-item"')

file_to_href = {
    "index.html": "/",
    "perceptron.html": "/perceptron",
    "perceptron_dashboard.html": "/perceptron",
    "backpropogation_dashboard.html": "/backpropogation",
    "mlp_dashboard.html": "/mlp",
    "activation_functions_dashboard.html": "/activation_functions",
    "digit_detection_dashboard.html": "/digit_detection",
    "cnn_dashboard.html": "/cnn",
    "opencv_dashboard.html": "/opencv_project",
    "rnn_dashboard.html": "/rnn_project",
    "sentiment_analysis_dashboard.html": "/sentiment_analysis"
}

for filename in files:
    filepath = os.path.join(template_dir, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        continue
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    new_nav = base_nav
    href = file_to_href.get(filename)
    if href:
        # add active class back
        new_nav = new_nav.replace(f'href="{href}" class="menu-item"', f'href="{href}" class="menu-item active"')
        
    new_content = re.sub(r'<nav class="sidebar-menu">.*?</nav>', new_nav, content, flags=re.DOTALL)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print(f"Updated {filename}")
