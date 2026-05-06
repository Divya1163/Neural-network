// Deep Learning Toolbox - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('Deep Learning Toolbox loaded');
    
    // Initialize menu item click handlers
    const menuItems = document.querySelectorAll('.menu-item');
    menuItems.forEach(item => {
        item.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            // Let the default navigation happen for links
            if (href && href !== '#') {
                window.location.href = href;
            }
        });
    });
    
    // Search functionality
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const query = e.target.value;
            if (query.length > 0) {
                console.log(`Searching for: ${query}`);
            }
        });
    }
    
    // Add model button
    const addModelBtn = document.querySelector('.btn-primary');
    if (addModelBtn) {
        addModelBtn.addEventListener('click', function() {
            console.log('Add Model clicked');
            // TODO: Open add model dialog
        });
    }
    
    // Import data button
    const importDataBtn = document.querySelector('.btn-secondary');
    if (importDataBtn) {
        importDataBtn.addEventListener('click', function() {
            console.log('Import Data clicked');
            // TODO: Open import dialog
        });
    }
    
    // Model card click
    const modelCards = document.querySelectorAll('.model-card');
    modelCards.forEach(card => {
        card.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href) {
                window.location.href = href;
            }
        });
    });
    
    // Icon buttons
    const iconBtns = document.querySelectorAll('.icon-btn');
    iconBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            console.log('Icon button clicked');
            // TODO: Add notification/message functionality
        });
    });
});


