document.getElementById('langToggle').addEventListener('click', function() {
    const dropdown = document.getElementById('langDropdown');
    dropdown.classList.toggle('hidden');
  });
  
  // Itxi dropdown-a kanpoan klik eginez gero
  document.addEventListener('click', function(event) {
    const dropdown = document.getElementById('langDropdown');
    const toggleButton = document.getElementById('langToggle');
    
    if (!dropdown.contains(event.target) && !toggleButton.contains(event.target)) {
      dropdown.classList.add('hidden');
    }
  });

let isAdvancedMode = false;

// Bisualki aurreratutako modua aktibatzeko/desaktibatzeko funtzioa
document.getElementById('advancedToggle').addEventListener('click', function () {
    isAdvancedMode = !isAdvancedMode;
    const icon = document.getElementById('advancedIcon');

    if (isAdvancedMode) {
        icon.classList.remove('text-[#D5D6DD]');
        icon.classList.add('text-[#607AFB]');
    } else {
        icon.classList.remove('text-[#607AFB]');
        icon.classList.add('text-[#D5D6DD]');
    }
});

// Formularioa atzematea eta egoera modu aurreratuan bidaltzea
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
        }).catch(error => console.error('Ezin izan da mezua bidali:', error));
    }
});