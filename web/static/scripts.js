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

let isAdvancedMode = false;  // Estado local del modo avanzado

// Función para activar/desactivar el modo avanzado visualmente
document.getElementById('advancedToggle').addEventListener('click', function () {
    isAdvancedMode = !isAdvancedMode;
    const icon = document.getElementById('advancedIcon');

    if (isAdvancedMode) {
        icon.classList.remove('text-[#D5D6DD]');
        icon.classList.add('text-[#607AFB]'); // Color azul cuando está activado
    } else {
        icon.classList.remove('text-[#607AFB]');
        icon.classList.add('text-[#D5D6DD]'); // Color gris cuando está desactivado
    }
});

// Interceptar el formulario y enviar el estado del modo avanzado
document.querySelector('form').addEventListener('submit', function (event) {
    event.preventDefault();

    const form = event.target;
    const inputField = form.querySelector('input[name="user_input"]');
    const userInput = inputField.value.trim();

    if (userInput) {
        const formData = new FormData(form);
        formData.append('advanced_mode', isAdvancedMode);

        fetch(form.action, {
            method: form.method,
            body: formData
        }).then(response => {
            if (response.ok) {
                window.location.reload();
            }
        }).catch(error => console.error('Error al enviar el mensaje:', error));
    }
});