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
    const submitButton = document.getElementById('submitButton');
    const submitText = document.getElementById('submitText');
    const loadingSpinner = document.getElementById('loadingSpinner');

    if (userInput) {
        // Spinner-a eta botoia kargatzen erakutsi
        submitText.classList.add('hidden');
        loadingSpinner.classList.remove('hidden');
        submitButton.disabled = true;
        submitButton.classList.add('opacity-75');

        const formData = new FormData(form);
        formData.append('advanced_mode', isAdvancedMode);

        fetch(form.action, {
            method: form.method,
            body: formData
        }).then(response => {
            if (response.ok) {
                // Input-a garbitu
                inputField.value = '';
                
                // Botoia eta spinner-a berrezarri
                submitText.classList.remove('hidden');
                loadingSpinner.classList.add('hidden');
                submitButton.disabled = false;
                submitButton.classList.remove('opacity-75');
                
                window.location.reload();
            }
        }).catch(error => {
            console.error(error);
            submitText.classList.remove('hidden');
            loadingSpinner.classList.add('hidden');
            submitButton.disabled = false;
            submitButton.classList.remove('opacity-75');
        });
    }
});

function handleFileSelect(event) {
    const files = event.target.files;
    const fileListDiv = document.getElementById('fileList');
    const translations = window.translations || {};

    if (files.length > 0) {
        for (let i = 0; i < files.length; i++) {
            fileListDiv.innerHTML += `
                <div class="flex items-center gap-2 mb-1">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="flex-shrink-0">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <path d="M14 2v6h6"/>
                        <path d="M12 18v-6"/>
                        <path d="M9 15h6"/>
                    </svg>
                    <span class="truncate">${files[i].name}</span>
                </div>
            `;
        }
        
        fileListDiv.classList.remove('hidden');
    } else {
        fileListDiv.classList.add('hidden');
    }
}