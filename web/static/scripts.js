document.getElementById('langToggle').addEventListener('click', function() {
    const dropdown = document.getElementById('langDropdown');
    dropdown.classList.toggle('hidden');
  });
  
  // Cerrar el dropdown si se hace clic fuera
  document.addEventListener('click', function(event) {
    const dropdown = document.getElementById('langDropdown');
    const toggleButton = document.getElementById('langToggle');
    
    if (!dropdown.contains(event.target) && !toggleButton.contains(event.target)) {
      dropdown.classList.add('hidden');
    }
  });